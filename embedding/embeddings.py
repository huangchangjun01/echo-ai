from typing import List, Sequence

from langchain.embeddings.base import Embeddings

try:
    from embedding import models as _repo_models
    _has_repo_text = hasattr(_repo_models, "compute_text_embedding")
    _has_repo_image = hasattr(_repo_models, "compute_embedding")
except Exception:
    _repo_models = None
    _has_repo_text = False
    _has_repo_image = False


def _sha256_fallback(texts: List[str]) -> List[List[float]]:
    """Lightweight deterministic fallback using SHA256 hash."""
    import hashlib
    dim = 384
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode('utf-8')).digest()
        vals = []
        i = 0
        while len(vals) < dim:
            b = h[i % len(h)]
            vals.append((b / 255.0) * 2.0 - 1.0)
            i += 1
        vecs.append(vals[:dim])
    return vecs


def _image_fallback(images: List) -> List[List[float]]:
    """Deterministic image fallback hashing bytes to a fixed-dim vector."""
    import hashlib
    import io
    dim = 384
    vecs = []
    for img in images:
        if isinstance(img, bytes):
            b = img
        else:
            try:
                from PIL import Image
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                b = buf.getvalue()
            except Exception:
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


class ChineseCLIPEmbeddings(Embeddings):
    """LangChain Embeddings wrapper.

    Uses project's compute_text_embedding / compute_embedding if available,
    otherwise falls back to sentence-transformers or lightweight SHA256 fallback.
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.device = device
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"

        if _has_repo_text:
            self._embed_fn = _repo_models.compute_text_embedding
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(self.model_name, device=self.device)
                self._embed_fn = lambda texts: self._st.encode(texts, convert_to_numpy=False).tolist()
            except Exception:
                self._st = None
                self._embed_fn = _sha256_fallback

        if _has_repo_image:
            self._image_embed_fn = _repo_models.compute_embedding
        else:
            self._image_embed_fn = _image_fallback

    def _normalize(self, res, dim: int = 384) -> List[List[float]]:
        """Normalize embedding result to List[List[float]]."""
        if res is None:
            return []
        try:
            res_list = res.tolist() if hasattr(res, "tolist") else res
        except Exception:
            res_list = res
        if isinstance(res_list, (list, tuple)) and len(res_list) > 0 and all(isinstance(x, (int, float)) for x in res_list):
            return [[float(x) for x in res_list]]
        if isinstance(res_list, (list, tuple)):
            out = []
            for item in res_list:
                if item is None:
                    continue
                try:
                    item_list = item.tolist() if hasattr(item, "tolist") else item
                except Exception:
                    item_list = list(item)
                if isinstance(item_list, (list, tuple)) and all(isinstance(x, (int, float)) for x in item_list):
                    out.append([float(x) for x in item_list])
                else:
                    try:
                        out.append([float(x) for x in item_list])
                    except Exception:
                        continue
            return out
        return []

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if _has_repo_text:
            try:
                return self._normalize(self._embed_fn(texts))
            except Exception:
                pass
            return self._normalize([self._embed_fn(t) for t in texts])
        return self._normalize(self._embed_fn(texts))

    def embed_query(self, text: str) -> List[float]:
        if _has_repo_text:
            for fn in (self._embed_fn, lambda t: self._embed_fn([t]), lambda t: self._embed_fn([t])):
                try:
                    vec = self._normalize(fn(text))
                    if vec:
                        return vec[0]
                except Exception:
                    pass
        vec = self._normalize(self._embed_fn([text]))
        return vec[0] if vec else []

    def embed_images(self, images: Sequence) -> List[List[float]]:
        if _has_repo_image:
            try:
                return self._normalize([self._image_embed_fn(img) for img in images])
            except Exception:
                pass
        return self._normalize(self._image_embed_fn(list(images)))

    def embed_image(self, image) -> List[float]:
        vecs = self.embed_images([image])
        return vecs[0] if vecs else []