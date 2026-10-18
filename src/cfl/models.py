"""Model zoo. Every model maps uniformised features -> expected uniformised next-month return."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def model_zoo(seed: int = 42) -> dict:
    return {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=40, max_features=0.5,
            n_jobs=-1, random_state=seed),
        "GBM": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=40,
            l2_regularization=1.0, random_state=seed),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3,
            max_iter=400, early_stopping=True, validation_fraction=0.2, random_state=seed),
    }


# Heuristic (non-ML) benchmark signals: column name -> sign (+1: high value = long)
BENCHMARKS = {
    "Momentum 12-1": ("mom_12_1", +1),
    "Hedging pressure": ("hp", -1),      # Keynes / Hirshleifer: long when hedgers are net short
    "Carry": ("carry", +1),
}
