from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.event_features import attribute_shap, explain_attributes
from src.models.utils import CLASS_NAMES

class BaseDisasterModel(ABC):
    """Common interface for all Disaster Auditor models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Short category (e.g. 'Tree Ensemble', 'Gradient Boosting', 'LLM Zero-Shot')."""
        pass

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference on a single feature record dictionary.
        Returns:
            {
                "predicted_id": int,
                "predicted_class": str,
                "confidence": float,
                "probabilities": Dict[str, float]
            }
        """
        pass

    @abstractmethod
    def explain(self, features: Dict[str, float], target_class_idx: Optional[int] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        Calculate or retrieve SHAP attributions for the record.
        Returns:
            {
                "top_positive": List[Dict],
                "top_negative": List[Dict],
                "top_all": List[Dict],
                "by_attribute": List[Dict],   # SHAP summed into the 6 raw attributes, with % shares
                "human_explanations": List[str]
            }
        """
        pass

    @staticmethod
    def _package_shap(feature_names: List[str], features: Dict[str, float], class_shap, pred_id: int,
                      top_k: int) -> Dict[str, Any]:
        """Shared by all models: per-feature top lists (for tools) plus the per-attribute view (for people)."""
        contributions = [{
            "feature": feat,
            "value": float(features.get(feat, 0.0)),
            "shap_value": round(float(s_val), 4),
            "direction": "positive" if s_val > 0 else "negative",
            "abs_impact": abs(float(s_val)),
        } for feat, s_val in zip(feature_names, class_shap)]
        ranked = sorted(contributions, key=lambda x: x["abs_impact"], reverse=True)
        target_name = CLASS_NAMES[pred_id]
        by_attribute = attribute_shap(feature_names, features, class_shap)
        return {
            "target_class": target_name,
            "target_id": pred_id,
            "top_positive": [c for c in ranked if c["direction"] == "positive"][:top_k],
            "top_negative": [c for c in ranked if c["direction"] == "negative"][:top_k],
            "top_all": ranked[:top_k],
            "by_attribute": by_attribute,
            "human_explanations": explain_attributes(by_attribute, target_name),
        }

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """
        Return headline test set metrics.
        Returns:
            {
                "macro_f1": float,
                "macro_f1_ci": [float, float],
                "accuracy": float,
                "weighted_f1": float,
                "per_class_f1": Dict[str, float],
                "per_class_recall": Dict[str, float]
            }
        """
        pass
