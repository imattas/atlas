"""Safe optional AtlasCTF ranker boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankRequest:
    """Rank request limited to strategies, features, and seed."""

    strategy_ids: tuple[str, ...]
    features: dict[str, float]
    seed: int = 0


@dataclass(frozen=True)
class RankResponse:
    """Allowed rank response fields."""

    ordered_strategy_ids: tuple[str, ...]
    budget_multipliers: dict[str, float]
    explanation: str


def rank(request: RankRequest) -> RankResponse:
    """Return a deterministic transparent baseline ranking."""
    if not request.features:
        raise ValueError("features are required")
    ordered = tuple(sorted(request.strategy_ids, key=lambda item: (len(item), item)))
    return RankResponse(
        ordered,
        {strategy: 1.0 + index * 0.1 for index, strategy in enumerate(ordered)},
        "python transparent baseline",
    )
