import os
import json
import random
import joblib
from typing import Dict, Any, List, Optional, Tuple
from src.models.rf_model import RandomForestModel
from src.models.lgb_model import LightGBMModel
from src.models.xgb_model import XGBoostModel
from src.event_features import build_features, explain_contrast, mark_filled
from src.scale_prior import PRIOR_PATH, ScalePrior

class ModelRegistry:
    def __init__(self, bank_path: str = "models/multi_model_bank.joblib", meta_path: str = "models/multi_model_metadata.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.class_names = self.metadata["class_names"]
        self.feature_names = self.metadata["feature_names"]
        self.leaderboard = self.metadata["leaderboard"]
        self.mcnemar = self.metadata["mcnemar"]

        # Load precomputed cases
        self.cases: List[Dict[str, Any]] = joblib.load(bank_path)
        self.case_map = {c["case_id"]: c for c in self.cases}

        # Lazy models container
        self._models: Dict[str, Any] = {}
        self._scale_prior: Optional[ScalePrior] = None

    @property
    def scale_prior(self) -> ScalePrior:
        if self._scale_prior is None:
            self._scale_prior = ScalePrior.load(PRIOR_PATH)
        return self._scale_prior

    def scale_mix(self, case: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """For an event whose magnitude scale was not given: P(scale | its country, region, month), else None."""
        if not case.get("scale_not_given"):
            return None
        event = case.get("event") or {}
        weights = self.scale_prior.weights(event.get("country"), event.get("region"), event.get("start_month"))
        # A scale the user ruled out ("ไม่ใช่" to the app's proposal) is not one of the possibilities.
        kept = {s: w for s, w in weights.items() if s not in (case.get("scale_ruled_out") or [])}
        total = sum(kept.values())
        return {s: w / total for s, w in kept.items()} if total else weights

    def predict_case(self, case: Dict[str, Any], model_name: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """One model's prediction on a case, plus the feature record to explain it with.

        Scale not given: the model runs once per likely scale and the class probabilities are averaged,
        weighted by how often each scale is recorded there (sum over s of P(type | ..., s) * P(s | place, month)).
        SHAP then explains the single scale that contributes most to the winning class."""
        model = self.get_model(model_name)
        mix = self.scale_mix(case)
        if not mix:
            return model.predict(case["features"]), case["features"]
        event = case["event"]
        runs = {s: build_features({**event, "magnitude_scale": s}, self.feature_names) for s in mix}
        preds = {s: model.predict(f) for s, f in runs.items()}
        probs = {c: sum(w * preds[s]["probabilities"][c] for s, w in mix.items()) for c in self.class_names}
        best = max(probs, key=probs.get)
        lead = max(mix, key=lambda s: mix[s] * preds[s]["probabilities"][best])
        prediction = {
            "predicted_id": self.class_names.index(best),
            "predicted_class": best,
            "confidence": probs[best],
            "probabilities": probs,
            "scale_mix": [{"scale": s, "weight": round(w, 3), "predicted_class": preds[s]["predicted_class"]}
                          for s, w in mix.items()],
            "explained_with_scale": lead,
        }
        return prediction, runs[lead]

    def get_model(self, name: str):
        if name not in self._models:
            if name == "Random Forest":
                self._models[name] = RandomForestModel()
            elif name == "LightGBM":
                self._models[name] = LightGBMModel()
            elif name == "XGBoost":
                self._models[name] = XGBoostModel()
            else:
                raise ValueError(f"Unknown model name: {name}")
        return self._models[name]

    @property
    def available_models(self) -> List[str]:
        return ["Random Forest", "LightGBM", "XGBoost"]

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.case_map.get(case_id)

    def get_random_case(
        self,
        disaster_type: Optional[str] = None,
        only_hard: bool = False,
        only_disagreement: bool = False,
        region: Optional[str] = None,
        misclassified_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        filtered = self.cases
        if disaster_type and disaster_type.lower() != "all":
            filtered = [c for c in filtered if c["ground_truth_class"].lower() == disaster_type.lower()]
        if only_hard:
            # Considered hard if ANY of the models failed
            filtered = [c for c in filtered if any(not ev["is_correct"] for ev in c["evaluations"].values())]
        if only_disagreement:
            filtered = [c for c in filtered if c.get("is_disagreement", False)]
        if region and region.lower() != "all":
            filtered = [c for c in filtered if c["summary"]["region"].lower() == region.lower()]
        if misclassified_by:
            filtered = [c for c in filtered if not c["evaluations"][misclassified_by]["is_correct"]]

        if not filtered:
            return None
        return random.choice(filtered)

    def audit_case(self, case: Dict[str, Any], model_names: List[str]) -> Dict[str, Any]:
        """
        Produce a unified multi-model audit package for a case.
        Uses precomputed evaluations for 0ms instant display, 
        and extracts SHAP explanations for selected models.
        """
        results = {}
        for m_name in model_names:
            features = case["features"]
            if m_name in case["evaluations"]:
                eval_info = case["evaluations"][m_name]
            else:
                # Live inference fallback (averaged over likely scales when the scale was not given)
                eval_info, features = self.predict_case(case, m_name)
                # A user-described event has no ground truth until the user confirms it: None = unverified.
                truth_id = case.get("ground_truth_id")
                eval_info["is_correct"] = None if truth_id is None else bool(eval_info["predicted_id"] == truth_id)

            # Explain
            model_obj = self.get_model(m_name)
            shap_info = model_obj.explain(features, target_class_idx=eval_info["predicted_id"])
            # Attributes a user left out were filled with training defaults: say so instead of citing them as reasons.
            mark_filled(shap_info, case.get("filled", []))

            results[m_name] = {
                "prediction": eval_info,
                "shap": shap_info,
                "is_correct": eval_info["is_correct"],
                "contrast": self.contrast(case, m_name, eval_info, shap_info, features),
            }

        return {
            "case": case,
            "models": results,
            "is_disagreement": case.get("is_disagreement", False)
        }

    def contrast(self, case: Dict[str, Any], model_name: str, prediction: Dict[str, Any],
                 predicted_shap: Dict[str, Any], features: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """'Why not X?': SHAP for the class the model did NOT pick, next to the one it did.

        X is the true type when the model got it wrong (the case worth auditing), otherwise the
        runner-up class (what it nearly said). None when the model is right and has no runner-up.
        """
        truth_id = case.get("ground_truth_id")
        if truth_id is not None and truth_id != prediction["predicted_id"]:
            target_id, reason = truth_id, "true type"
        else:
            ranked = sorted(prediction.get("probabilities", {}).items(), key=lambda kv: kv[1], reverse=True)
            runner_up = next((c for c, _ in ranked if c != prediction["predicted_class"]), None)
            if runner_up is None:
                return None
            target_id, reason = self.class_names.index(runner_up), "runner-up"
        shap_info = self.get_model(model_name).explain(features or case["features"], target_class_idx=target_id)
        mark_filled(shap_info, case.get("filled", []))
        result = {
            "class": shap_info["target_class"],
            "reason": reason,
            "probability": prediction.get("probabilities", {}).get(shap_info["target_class"]),
            "by_attribute": shap_info["by_attribute"],
        }
        result["human_explanations"] = explain_contrast(predicted_shap, result)
        return result

    def refresh_contrast(self, audit: Dict[str, Any]) -> None:
        """After the user confirms the true type, 'why not X?' should ask about that type."""
        for m_name, m in audit["models"].items():
            features = self.explained_features(audit["case"], m["prediction"])
            m["contrast"] = self.contrast(audit["case"], m_name, m["prediction"], m["shap"], features)

    def explained_features(self, case: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, float]:
        """The feature record a prediction was explained with (its lead scale when the scale was not given)."""
        scale = prediction.get("explained_with_scale")
        if not scale:
            return case["features"]
        return build_features({**case["event"], "magnitude_scale": scale}, self.feature_names)

    def country_regions(self) -> Dict[str, str]:
        """Country -> region, as recorded in the test bank (used to fill in a described event's region)."""
        return {c["summary"]["country"]: c["summary"]["region"] for c in self.cases
                if c["summary"]["region"] != "Unknown"}
