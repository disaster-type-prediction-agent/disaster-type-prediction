"""How often each magnitude scale is recorded for a place and month: P(scale | country, month).

Used when a user describes an event but never says what was measured. Filling the gap with the "Unknown"
scale would tell the models "nothing was measured", which in EM-DAT means Landslide or Epidemic. Instead
the app runs the models once per scale and averages the answers, weighted by how often each scale is
recorded for that country and month in the TRAINING split (the test split stays unseen).

Counts back off from country+month to country, region+month, region and the whole training set, each level
shrunk toward the one above with SMOOTHING pseudo-counts, so a country with few records still gets sensible
weights. Build the table with:  python -m src.scale_prior
"""
import json
from collections import defaultdict
from typing import Dict, Optional

from src.event_features import MAGNITUDE_SCALES

PRIOR_PATH = "models/scale_prior.json"
DATA_PATH = "data/processed/disaster_processed_feature_selected.csv"
SMOOTHING = 5.0  # pseudo-counts borrowed from the parent level
MIN_WEIGHT = 0.01  # scales below this share are dropped (and the rest renormalized)


def build_prior(data_path: str = DATA_PATH) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """Scale counts per (country, month) and (region, month) on the same training split as the models."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(data_path)
    X, y = data.drop(columns=["disaster_type_encoded"]), data["disaster_type_encoded"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)  # as in export_all_models

    def one_hot(prefix):
        cols = [c for c in X_train.columns if c.startswith(prefix)]
        return X_train[cols].idxmax(axis=1).str[len(prefix):].where(X_train[cols].max(axis=1) > 0)

    frame = pd.DataFrame({"country": one_hot("country_"), "region": one_hot("region_"),
                          "scale": one_hot("magnitude_scale_"), "month": X_train["start_month"].astype(int)})
    table = {"country": defaultdict(lambda: defaultdict(dict)), "region": defaultdict(lambda: defaultdict(dict))}
    for level in table:
        grouped = frame.dropna(subset=[level, "scale"]).groupby([level, "month", "scale"]).size()
        for (key, month, scale), n in grouped.items():
            table[level][key][str(month)][scale] = int(n)
    return {level: {k: dict(v) for k, v in rows.items()} for level, rows in table.items()}


def _sum(by_month: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    total: Dict[str, int] = defaultdict(int)
    for counts in by_month.values():
        for scale, n in counts.items():
            total[scale] += n
    return total


def _shrink(counts: Dict[str, int], parent: Dict[str, float]) -> Dict[str, float]:
    n = sum(counts.values())
    return {s: (counts.get(s, 0) + SMOOTHING * parent[s]) / (n + SMOOTHING) for s in MAGNITUDE_SCALES}


class ScalePrior:
    def __init__(self, table: Dict[str, Dict[str, Dict[str, Dict[str, int]]]]):
        self.table = table
        everything = _sum({k: _sum(v) for k, v in table["region"].items()})
        n = sum(everything.values())
        self.base = {s: everything.get(s, 0) / n for s in MAGNITUDE_SCALES}

    @classmethod
    def load(cls, path: str = PRIOR_PATH) -> "ScalePrior":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def weights(self, country: Optional[str] = None, region: Optional[str] = None,
                month: Optional[int] = None) -> Dict[str, float]:
        """P(scale | what is known of the place and month), largest first; tiny shares dropped."""
        p = self.base
        month_key = str(int(month)) if month and 1 <= int(month) <= 12 else None
        for level, key in (("region", region), ("country", country)):
            rows = self.table[level].get(key) if key else None
            if not rows:
                continue
            p = _shrink(_sum(rows), p)
            if month_key and month_key in rows:
                p = _shrink(rows[month_key], p)
        kept = {s: w for s, w in p.items() if w >= MIN_WEIGHT}
        total = sum(kept.values())
        return dict(sorted(((s, w / total) for s, w in kept.items()), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    prior = build_prior()
    with open(PRIOR_PATH, "w", encoding="utf-8") as f:
        json.dump(prior, f, ensure_ascii=False, sort_keys=True)
    print(f"Wrote {PRIOR_PATH}: {len(prior['country'])} countries, {len(prior['region'])} regions")
