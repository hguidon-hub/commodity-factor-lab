"""Model zoo. Every model maps uniformised features -> expected uniformised next-month return.

A zoo entry is either an estimator (uses all features) or a tuple (estimator, feature_list)
— the latter is how the ablation "price only" vs "positioning only" is run.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def _ridge():
    return Ridge(alpha=10.0)


def _gbm(seed: int = 42):
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed)


def model_zoo(seed: int = 42) -> dict:
    """Main models, trained on every available feature."""
    return {
        "Ridge": _ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=40, max_features=0.5,
            n_jobs=-1, random_state=seed),
        "GBM": _gbm(seed),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3,
            max_iter=400, early_stopping=True, validation_fraction=0.2, random_state=seed),
    }


def ablation_zoo(price_feats: list[str], pos_feats: list[str], seed: int = 42) -> dict:
    """Same Ridge and GBM, restricted to one feature group: isolates what positioning adds."""
    zoo = {}
    if price_feats:
        zoo["Ridge [price only]"] = (_ridge(), price_feats)
        zoo["GBM [price only]"] = (_gbm(seed), price_feats)
    if pos_feats:
        zoo["Ridge [positioning only]"] = (_ridge(), pos_feats)
        zoo["GBM [positioning only]"] = (_gbm(seed), pos_feats)
    return zoo


def selection_zoo(ks: tuple[int, ...] = (6,)) -> dict:
    """Ridge on the k signals with the best track record in each training window (no look-ahead).

    Several k are run so that the result can be checked for sensitivity to this choice.
    """
    return {f"Ridge [selected {k}]": {"est": _ridge(), "select_k": k} for k in ks}


MAIN_MODELS = ["Ridge", "RandomForest", "GBM", "MLP"]

# Heuristic (non-ML) benchmark signals: name -> (column, sign) ; +1: high value = long
BENCHMARKS = {
    "Momentum 12-1": ("mom_12_1", +1),
    "Hedging pressure": ("hp", -1),           # Keynes / Basu-Miffre: long when hedgers are net short
    "Managed money contrarian": ("mm_crowding", -1),   # fade crowded speculative positioning
    "Carry": ("carry", +1),
}
