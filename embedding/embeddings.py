from typing import List, Sequence, Optional, Dict, Any

from langchain.embeddings.base import Embeddings

try:
    from embedding import models as _repo_models
    _has_repo_text = hasattr(_repo_models, "compute_text_embedding")
    _has_repo_image = hasattr(_repo_models, "compute_embedding")
except Exception:
    _repo_models = None
    _has_repo_text = False
    _has_repo_image = False


class ChineseCLIPEmbeddings(Embeddings):
    """LangChain Embeddings wrapper.

    If the project already provides compute_text_embedding(text)->List[float], use it.
    Otherwise fall back to sentence-transformers all-MiniLM for CPU-friendly embeddings.

    Additionally this class exposes image embedding helpers `embed_image` and `embed_images`.
    If the project provides `compute_embedding(image_bytes)->List[float]` it will be used; otherwise
    a deterministic fallback (SHA256-derived vector) will be returned so the application remains runnable.
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"

        if _has_repo_text:
            self._embed_fn = _repo_models.compute_text_embedding
            self._is_repo = True
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(self.model_name, device=self.device)
                self._embed_fn = lambda texts: self._st.encode(texts, convert_to_numpy=False).tolist()
            except Exception:
                # Lightweight deterministic fallback when sentence-transformers is unavailable.
                # Produces stable pseudo-embeddings derived from SHA256 hash of the text.
                import hashlib

                self._st = None

                def _fallback(texts):
                    vecs = []
                    dim = 384
                    for t in texts:
                        h = hashlib.sha256(t.encode('utf-8')).digest()
                        # Expand hash to required dim by repeating and converting bytes to floats in [-1,1]
                        vals = []
                        i = 0
                        while len(vals) < dim:
                            b = h[i % len(h)]
                            # convert byte to float in range [-1,1]
                            vals.append((b / 255.0) * 2.0 - 1.0)
                            i += 1
                        vecs.append(vals[:dim])
                    return vecs

                self._embed_fn = _fallback
            self._is_repo = False

        # Image embedding support: use repo compute_embedding if available, else provide a fallback
        if _has_repo_image:
            self._image_embed_fn = _repo_models.compute_embedding
            self._is_repo_image = True
        else:
            # simple deterministic image fallback that hashes image bytes
            import hashlib
            import io

            def _image_fallback(images):
                vecs = []
                dim = 384
                for img in images:
                    # Accept either bytes or PIL Image-like (PIL not required here)
                    if isinstance(img, bytes):
                        b = img
                    else:
                        # try to convert PIL Image-like to bytes
                        try:
                            from PIL import Image

                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            b = buf.getvalue()
                        except Exception:
                            # fallback to string representation
                            b = str(img).encode('utf-8')
                    h = hashlib.sha256(b).digest()
                    vals = []
                    i = 0
                    while len(vals) < dim:
                        byte = h[i % len(h)]
                        vals.append((byte / 255.0) * 2.0 - 1.0)
                        i += 1
                    vecs.append(vals[:dim])
                return vecs

            self._image_embed_fn = _image_fallback
            self._is_repo_image = False

    # --- Helpers to normalize returned shapes ---
    def _normalize_batch(self, res) -> List[List[float]]:
        """Ensure the result is a list of vectors (List[List[float]]).

        Accepts numpy arrays, single vector (list[float]) or batch (list[list[float]]).
        """
        if res is None:
            return []

        # If numpy-like
        try:
            # numpy arrays have tolist()
            if hasattr(res, "tolist"):
                res_list = res.tolist()
            else:
                res_list = res
        except Exception:
            res_list = res

        # If single vector (flat list of numbers), wrap it
        if isinstance(res_list, (list, tuple)) and len(res_list) > 0 and all(isinstance(x, (int, float)) for x in res_list):
            return [[float(x) for x in res_list]]

        # If batch-like
        if isinstance(res_list, (list, tuple)):
            out = []
            for item in res_list:
                if item is None:
                    continue
                if hasattr(item, "tolist"):
                    item_list = item.tolist()
                else:
                    item_list = item
                if isinstance(item_list, (list, tuple)) and all(isinstance(x, (int, float)) for x in item_list):
                    out.append([float(x) for x in item_list])
                else:
                    # last resort: try to coerce
                    try:
                        out.append([float(x) for x in list(item_list)])
                    except Exception:
                        continue
            return out

        # Unknown type -> return empty
        return []

    def _normalize_vector(self, res) -> List[float]:
        """Return a single vector (list[float]) from various possible return shapes."""
        batch = self._normalize_batch(res)
        return batch[0] if batch else []

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if self._is_repo:
            # Try batch call first (many repo implementations support batch)
            try:
                res = self._embed_fn(texts)
                normalized = self._normalize_batch(res)
                if normalized:
                    return normalized
            except Exception:
                pass

            # Fallback to per-item calls
            out = []
            for t in texts:
                try:
                    r = self._embed_fn(t)
                except Exception:
                    r = self._embed_fn([t]) if not isinstance(t, (list, tuple)) else self._embed_fn(t)
                out.append(self._normalize_vector(r))
            return out

        else:
            return self._normalize_batch(self._embed_fn(texts))

    def embed_query(self, text: str) -> List[float]:
        if self._is_repo:
            # Try direct call
            try:
                res = self._embed_fn(text)
                vec = self._normalize_vector(res)
                if vec:
                    return vec
            except Exception:
                pass

            # Try batch-style call
            try:
                res = self._embed_fn([text])
                vec = self._normalize_vector(res)
                if vec:
                    return vec
            except Exception:
                pass

            # Final fallback
            return self._normalize_vector(self._embed_fn([text]))
        else:
            return self._normalize_vector(self._embed_fn([text]))

    # New image embedding helpers
    def embed_images(self, images: Sequence) -> List[List[float]]:
        """Embed a sequence of images.

        Each item in `images` may be raw bytes or a PIL.Image-like object. Returns a list of vectors.
        If a repo-level `compute_embedding` exists it will be used; otherwise a deterministic fallback is used.
        """
        images = list(images)
        # If repo implementation exists, try per-item calls (repo compute_embedding expects single bytes, not list)
        if self._is_repo_image:
            try:
                out = []
                for img in images:
                    r = self._image_embed_fn(img)
                    out.append(self._normalize_vector(r))
                return out
            except Exception:
                pass

        # Fallback: ensure input is list-like and call fallback
        return self._normalize_batch(self._image_embed_fn(images))

    def embed_image(self, image) -> List[float]:
        """Embed a single image and return a 1-D vector."""
        vecs = self.embed_images([image])
        return vecs[0] if vecs else []

