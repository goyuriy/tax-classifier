import re
import pickle
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from src.utils import setup_logger

logger = setup_logger(__name__)

class HeuristicClassifier:
    """
    Classifies documents based on regex patterns in the text.
    Real PDFs often have the form number/name at the top, so we only consider matches
    in the first `header_chars` (header zone). Earliest match in that zone wins.
    Body text (e.g. "See Schedule 1" in instructions) is ignored.
    """
    DEFAULT_HEADER_CHARS = 50

    def __init__(self, header_chars: int = DEFAULT_HEADER_CHARS):
        self.header_chars = header_chars
        # Main 1040: specific phrase (avoids Schedule A (Form 1040) being tagged as 1040f)
        self._main_1040_pattern = re.compile(r"U\.S\.\s*Individual\s*Income\s*Tax\s*Return", re.IGNORECASE)
        # All other forms: earliest match in header zone wins
        self.patterns = [
            ("f1040sa", [r"Schedule\s+A\b", r"Schedule\s+A\s*\("]),
            ("f1040sb", [r"Schedule\s+B\b", r"Schedule\s+B\s*\("]),
            ("f1040sc", [r"Schedule\s+C\b", r"Schedule\s+C\s*\("]),
            ("f1040sd", [r"Schedule\s+D\b", r"Schedule\s+D\s*\("]),
            ("f1040se", [r"Schedule\s+E\b", r"Schedule\s+E\s*\("]),
            ("f1040s1", [r"Schedule\s+1\b", r"Schedule\s+1\s*\("]),
            ("f1040s2", [r"Schedule\s+2\b", r"Schedule\s+2\s*\("]),
            ("f1040s3", [r"Schedule\s+3\b", r"Schedule\s+3\s*\("]),
            ("f8949", [r"Form\s+8949", r"\b8949\b"]),
            ("f8889", [r"Form\s+8889", r"\b8889\b"]),
            ("f8863", [r"Form\s+8863"]),
            ("f8812", [r"Form\s+8812"]),
            ("f2441", [r"Form\s+2441"]),
            ("1040f", [r"Form\s+1040\b", r"Form\s+1040\s", r"\b1040\b"]),
        ]

    def _earliest_match_in_zone(self, text: str, end_pos: int) -> Optional[tuple[int, str]]:
        """Return (position, doc_type) for the pattern that matches earliest within text[:end_pos], or None."""
        zone = text[:end_pos]
        best_pos: Optional[int] = None
        best_type: Optional[str] = None
        for doc_type, regexes in self.patterns:
            for pattern in regexes:
                m = re.search(pattern, zone, re.IGNORECASE)
                if m and (best_pos is None or m.start() < best_pos):
                    best_pos = m.start()
                    best_type = doc_type
        if best_pos is not None and best_type is not None:
            return (best_pos, best_type)
        return None

    def predict(self, text: str) -> Optional[str]:
        """
        Returns the document type if a strong match is found in the header zone, else None.
        Header = first header_chars of the page (form ID is often at the top of real PDFs).
        """
        header = text[: self.header_chars]
        if self._main_1040_pattern.search(header):
            return "1040f"
        result = self._earliest_match_in_zone(text, self.header_chars)
        return result[1] if result else None

class SemanticClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifies documents using Embeddings + Logistic Regression.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.encoder = None
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.is_fitted = False

    @property
    def classes_(self) -> np.ndarray:
        """Class labels (order matches columns of predict_proba). Available after fit."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.classifier.classes_

    def _get_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def fit(self, X_text: List[str], y: List[str]):
        logger.info("Encoding training data...")
        encoder = self._get_encoder()
        embeddings = encoder.encode(X_text, show_progress_bar=True)
        
        logger.info("Training classifier...")
        self.classifier.fit(embeddings, y)
        self.is_fitted = True
        return self

    def predict(self, X_text: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        encoder = self._get_encoder()
        embeddings = encoder.encode(X_text, show_progress_bar=False)
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, X_text: List[str]) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        encoder = self._get_encoder()
        embeddings = encoder.encode(X_text, show_progress_bar=False)
        return self.classifier.predict_proba(embeddings)

    def predict_with_threshold(
        self,
        X_text: List[str],
        threshold: float = 0.7,
        low_confidence_label: str = "manual_review_needed",
    ) -> List[str]:
        """
        Predict labels; samples with max probability below threshold get low_confidence_label.
        """
        probs = self.predict_proba(X_text)
        return [
            self.classes_[np.argmax(p)] if np.max(p) > threshold else low_confidence_label
            for p in probs
        ]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: str):
        # Map MPS (Apple) device to CPU so checkpoints saved on Mac load on Linux/containers
        old_restore = getattr(
            torch.serialization, "default_restore_location", None
        )
        if old_restore is not None:

            def _cpu_restore(storage, location):
                if isinstance(location, str) and location.startswith("mps"):
                    location = "cpu"
                elif isinstance(location, torch.device) and location.type == "mps":
                    location = torch.device("cpu")
                return old_restore(storage, location)

            torch.serialization.default_restore_location = _cpu_restore
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        finally:
            if old_restore is not None:
                torch.serialization.default_restore_location = old_restore

class HybridClassifier:
    """
    Combines Heuristics and Semantic classification.
    Both use only the header zone (first max_chars) to avoid false positives from body text references.
    """
    DEFAULT_MAX_CHARS = 50

    def __init__(
        self,
        semantic_classifier: Optional[SemanticClassifier] = None,
        confidence_threshold: Optional[float] = None,
        low_confidence_label: str = "manual_review_needed",
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        self.heuristic = HeuristicClassifier(header_chars=max_chars)
        self.semantic = semantic_classifier
        self.confidence_threshold = confidence_threshold
        self.low_confidence_label = low_confidence_label
        self.max_chars = max_chars

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts document type using heuristics first, then semantic model.
        Both use only the first max_chars to avoid false positives from body text references.
        Returns dictionary with predicted label, source ('heuristic' or 'semantic'), and confidence.
        If confidence_threshold is set and semantic confidence is below it, label is low_confidence_label.
        """
        # Truncate to header zone (body text has references like "See Schedule A")
        header = text[:self.max_chars]

        # 1. Try Heuristics
        heuristic_label = self.heuristic.predict(header)
        if heuristic_label:
            return {
                "label": heuristic_label,
                "source": "heuristic",
                "confidence": 1.0
            }
        
        # 2. Fallback to Semantic
        if self.semantic and self.semantic.is_fitted:
            probs = self.semantic.predict_proba([header])[0]
            confidence = float(np.max(probs))
            semantic_label = self.semantic.classes_[np.argmax(probs)]

            if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                return {
                    "label": self.low_confidence_label,
                    "source": "semantic",
                    "confidence": confidence
                }
            return {
                "label": semantic_label,
                "source": "semantic",
                "confidence": confidence
            }
            
        return {
            "label": "unknown",
            "source": "none",
            "confidence": 0.0
        }


class SemanticOnlyClassifier:
    """
    Wraps SemanticClassifier to match the pipeline interface: predict(text: str) -> Dict.
    Uses only the first max_chars of text (header zone) to avoid false positives from body references.
    """
    DEFAULT_MAX_CHARS = 200

    def __init__(
        self,
        semantic_classifier: SemanticClassifier,
        max_chars: int = DEFAULT_MAX_CHARS,
    ):
        self.semantic = semantic_classifier
        self.max_chars = max_chars

    def predict(self, text: str) -> Dict[str, Any]:
        header = text[: self.max_chars]
        if not self.semantic.is_fitted:
            return {"label": "unknown", "source": "semantic", "confidence": 0.0}
        probs = self.semantic.predict_proba([header])[0]
        confidence = float(np.max(probs))
        label = self.semantic.classes_[np.argmax(probs)]
        
        # Filter out "other" class (background noise)
        if label == "other":
            return {
                "label": "unknown",
                "source": "semantic", 
                "confidence": confidence
            }
            
        return {
            "label": label,
            "source": "semantic",
            "confidence": confidence,
        }
