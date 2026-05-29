import os
import tempfile
import logging
from typing import Any, Dict, Optional

from config.config import QINIU_BASE_URL

logger = logging.getLogger(__name__)

# Local temp directory for downloaded files
TEMP_FILE_DIR = tempfile.gettempdir()

try:
	# import the downloader exported by utils package
	from utils import download_file
except Exception:
	# fallback simple implementation
	import requests

	def download_file(url: str) -> bytes:
		resp = requests.get(url, timeout=30)
		resp.raise_for_status()
		return resp.content


def ingest_background(user_id: str, file_obj: Dict[str, Any], embedding_model: Any, vectorstore: Any):
	"""Background worker: download file, compute embedding, and store into vector DB.

	This function is designed to be scheduled with FastAPI BackgroundTasks and thus should be
	synchronous (not async). It expects an `embedding_model` that implements
	`embed_documents` and `embed_images` similar to `ChineseCLIPEmbeddings`.

	Returns:
		Dict with 'success' status and message, or 'error' status with error details.
	"""

	try:
		file_id = file_obj.get('fileId')
		file_name = file_obj.get('fileName')
		file_key = file_obj.get('fileKey')
		url = file_obj.get('url')

		if not url and file_key and QINIU_BASE_URL:
			url = QINIU_BASE_URL.rstrip('/') + '/' + file_key.lstrip('/')
		if not url:
			logger.error(f"[ingest_background] Missing URL for file_id={file_id}, file_key={file_key}")
			return {"success": False, "error": "Missing URL", "file_id": file_id}

		logger.info(f"[ingest_background] Starting processing for file_id={file_id}, file_name={file_name}, url={url}")

		# download bytes
		try:
			data = download_file(url)
			logger.info(f"[ingest_background] Downloaded {len(data)} bytes for file_id={file_id}")
		except Exception as e:
			logger.error(f"[ingest_background] Failed to download file_id={file_id}, url={url}: {e}")
			return {"success": False, "error": f"Download failed: {str(e)}", "file_id": file_id}

		# Save downloaded data to local temp file for cleanup tracking
		try:
			_, ext = os.path.splitext(file_name or "")
			ext = (ext or "").lower()
			# Generate unique temp file name
			temp_fd, local_file_path = tempfile.mkstemp(suffix=ext, dir=TEMP_FILE_DIR)
			with os.fdopen(temp_fd, 'wb') as f:
				f.write(data)
			logger.info(f"[ingest_background] Saved temp file: {local_file_path}")
		except Exception as e:
			logger.warning(f"[ingest_background] Failed to save temp file for file_id={file_id}: {e}")
			local_file_path = None

		text_ext = {'.txt', '.md', '.csv', '.json', '.log', '.py', '.mdown', '.markdown'}
		image_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
		audio_ext = {'.mp3', '.wav', '.m4a', '.flac', '.aac'}
		video_ext = {'.mp4', '.mov', '.avi', '.mkv'}

		text_to_store = ""
		embeddings = None

		# Ensure we have an embedding model; if not, try to lazily import one
		if embedding_model is None:
			try:
				from embedding.embeddings import ChineseCLIPEmbeddings

				embedding_model = ChineseCLIPEmbeddings()
				logger.info(f"[ingest_background] Lazily initialized embedding model for file_id={file_id}")
			except Exception as e:
				logger.error(f"[ingest_background] Failed to initialize embedding model: {e}")
				embedding_model = None

		try:
			if ext in text_ext:
				try:
					text_to_store = data.decode('utf-8')
				except Exception:
					try:
						text_to_store = data.decode('gbk')
					except Exception:
						text_to_store = file_name or ""
				if embedding_model is not None:
					embeddings = embedding_model.embed_documents([text_to_store])
					logger.info(f"[ingest_background] Generated text embeddings for file_id={file_id}")
			elif ext in image_ext:
				text_to_store = file_name or ""
				if embedding_model is not None:
					embeddings = embedding_model.embed_images([data])
					logger.info(f"[ingest_background] Generated image embeddings for file_id={file_id}")
			elif ext in audio_ext or ext in video_ext:
				# For audio/video, we don't have dedicated encoders here. Fall back to image-like embedding of bytes
				text_to_store = file_name or ""
				if embedding_model is not None:
					embeddings = embedding_model.embed_images([data])
					logger.info(f"[ingest_background] Generated media embeddings for file_id={file_id}")
			else:
				# unknown type: store file name and attempt an image embed fallback
				text_to_store = file_name or ""
				if embedding_model is not None:
					embeddings = embedding_model.embed_images([data])
					logger.info(f"[ingest_background] Generated fallback embeddings for file_id={file_id}")
		except Exception as e:
			logger.error(f"[ingest_background] Embedding generation failed for file_id={file_id}: {e}")
			embeddings = None

		# Persist to vector store
		try:
			if vectorstore is not None:
				metadata = {
					'fileId': file_id,
					'fileName': file_name,
					'userId': user_id,
					'sourceUrl': url,
				}
				vectorstore.add_texts(ids=[file_id], texts=[text_to_store], metadatas=[metadata], embeddings=embeddings)
				logger.info(f"[ingest_background] Stored vector for file_id={file_id}")
		except Exception as e:
			logger.error(f"[ingest_background] Vector store failed for file_id={file_id}: {e}")
			return {"success": False, "error": f"Vector store failed: {str(e)}", "file_id": file_id}

		# Delete local temp file after successful vector generation
		if local_file_path:
			try:
				os.remove(local_file_path)
				logger.info(f"[ingest_background] Deleted temp file: {local_file_path}")
			except Exception as e:
				logger.warning(f"[ingest_background] Failed to delete temp file {local_file_path}: {e}")

		logger.info(f"[ingest_background] Successfully processed file_id={file_id}")
		return {"success": True, "file_id": file_id}

	except Exception as e:
		logger.error(f"[ingest_background] Unexpected error: {e}")
		return {"success": False, "error": f"Unexpected error: {str(e)}", "file_id": file_obj.get('fileId')}