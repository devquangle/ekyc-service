import numpy as np
from typing import Tuple, Optional, List
from utils.logger import logger


class FaceEmbeddingService:
    """
    Service for extracting, validating, and L2-normalizing ArcFace deep learning embeddings (512-d).
    Directly interfaces with InsightFace recognition model on standardized 112x112 aligned faces.
    Calculates cosine similarity with strict dimension and zero-norm verification.
    """

    def __init__(self, face_app=None):
        self.face_app = face_app

    def extract_embedding(
        self, aligned_face: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, float, List[str]]:
        """
        Extracts 512-d normalized embedding vector from a standardized 112x112 aligned face image.

        Args:
            aligned_face: 112x112 BGR aligned face image.

        Returns:
            Tuple: (normalized_vector, dimension, norm, errors)
        """
        errors: List[str] = []

        if aligned_face is None or aligned_face.size == 0:
            errors.append("FACE_EMBEDDING_FAILED")
            return None, 0, 0.0, errors

        raw_vec: Optional[np.ndarray] = None

        if self.face_app is not None:
            try:
                # 1. Primary: Direct recognition model inference on 112x112 aligned crop
                if hasattr(self.face_app, 'models') and 'recognition' in self.face_app.models:
                    rec_model = self.face_app.models['recognition']
                    raw_vec = rec_model.get_feat(aligned_face)
                # 2. Secondary: Direct face_app embedding method
                elif hasattr(self.face_app, 'get_embedding'):
                    raw_vec = self.face_app.get_embedding(aligned_face)
                # 3. Fallback: Full detection + recognition pipeline
                else:
                    faces = self.face_app.get(aligned_face)
                    if faces and hasattr(faces[0], 'embedding'):
                        raw_vec = faces[0].embedding
            except Exception as e:
                logger.error(f"[FACE_EMBEDDING] InsightFace recognition model inference error: {str(e)}")

        if raw_vec is None:
            logger.error("[FACE_EMBEDDING] Could not extract embedding from face image (No vector produced).")
            errors.append("FACE_EMBEDDING_FAILED")
            return None, 0, 0.0, errors

        # Flatten vector to 1D
        raw_vec = np.squeeze(np.asarray(raw_vec, dtype=np.float32))

        # Check for NaN / Inf
        if np.isnan(raw_vec).any() or np.isinf(raw_vec).any():
            logger.error("[FACE_EMBEDDING] Vector contains NaN or Inf values.")
            errors.append("INVALID_EMBEDDING")
            return None, len(raw_vec), 0.0, errors

        dimension = int(len(raw_vec))
        raw_norm = float(np.linalg.norm(raw_vec))

        if raw_norm == 0.0 or raw_norm < 1e-12:
            logger.error("[FACE_EMBEDDING] Embedding vector has zero norm.")
            errors.append("ZERO_NORM_EMBEDDING")
            return None, dimension, 0.0, errors

        # Standard L2 Unit Normalization (||v||_2 = 1.0)
        normalized_vec = (raw_vec / raw_norm).astype(np.float32)
        norm_after = float(np.linalg.norm(normalized_vec))

        return normalized_vec, dimension, norm_after, errors

    def calculate_cosine_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        Calculates Cosine Similarity between two L2-normalized embedding vectors.
        For unit vectors, Cosine Similarity = dot_product(emb1, emb2).

        Returns:
            Tuple: (similarity_score, errors)
        """
        errors: List[str] = []

        if emb1 is None or emb2 is None:
            errors.append("FACE_EMBEDDING_FAILED")
            return 0.0, errors

        dim1, dim2 = len(emb1), len(emb2)
        if dim1 != dim2:
            logger.error(f"[FACE_EMBEDDING] Dimension mismatch: {dim1} vs {dim2}")
            errors.append("EMBEDDING_DIMENSION_MISMATCH")
            return 0.0, errors

        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))

        if norm1 < 1e-12 or norm2 < 1e-12:
            errors.append("ZERO_NORM_EMBEDDING")
            return 0.0, errors

        # Compute dot product and clip into [-1.0, 1.0]
        dot_product = float(np.dot(emb1, emb2))
        similarity = dot_product / (norm1 * norm2)
        similarity = max(-1.0, min(1.0, similarity))

        return similarity, errors
