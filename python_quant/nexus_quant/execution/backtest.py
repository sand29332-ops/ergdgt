"""Batch execution backtest over an ``OrderBookEnv`` (plan_2.md Phase 2).

The env IS the synthetic tape + injectable book. ``run_backtest`` runs any
policy (callable obs→action, e.g. a baseline class or a loaded PPO policy)
over ``n_episodes`` from a seeded env factory, collects the honest metric set
(IS vs arrival, slippage vs **market** VWAP, fill/completion, MDD), batches a
per-regime breakdown, and ``summarize`` prints the comparison table — the E7
harness skeleton.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..envs.order_book_env import OrderBookEnv
from .metrics import (
    completion_rate,
    fill_rate,
    implementation_shortfall,
    max_drawdown,
    vwap_slippage,
)

Action = float
Policy = Callable[[OrderBookEnv], Action]  # obs-aware if needed; env is the tape


def _default_policy(env: OrderBookEnv) -> float:
    """Deterministic TWAP-style child: always post at the (near-)mid action."""
    return 0.0


def run_backtest(
    env_factories: Sequence[tuple[str, Callable[[], OrderBookEnv]]]
    | tuple[str, Callable[[], OrderBookEnv]]
    | Callable[[], OrderBookEnv],
    *,
    policy: Policy | None = None,
    n_episodes: int = 50,
    seed: int = 0x51ED,
    name: str = "policy",
) -> list[dict]:
    """Run ``policy`` over each (regime, env factory) and return per-episode rows.

    ``env_factories`` is one of:
      * a single ``Callable[[], OrderBookEnv]`` → one regime label ``"default"``;
      * a ``(name, factory)`` tuple → one regime;
      * a sequence of ``(name, factory)`` tuples → per-regime breakdown.

    Each row: ``{"regime", "policy", "reward", "is_bps", "vwap_slip_bps",
    "fill_rate", "completion_rate", "leftover", "mdd_ticks", "steps"}``.
    ``is_bps`` uses the arrival mid (observable at t); ``vwap_slip_bps`` is vs
    the **market** VWAP the env tracks from exogenous flow (not self-execution).
    Both slippage columns are **pre-cost**: fee/rebate/impact money terms are
    folded into the env's reward calculation via the cost knobs, not into
    slippage — so every strategy is compared on the same timing-skill metric,
    and cost competitiveness is read from ``reward`` when the env is
    constructed with costs on.

    ``mdd_ticks`` retains the historical field name but is measured in
    price-tick × share units: maximum peak-to-trough **gross**
    mark-to-market PnL. The reset PnL is the first path point, so a first-step
    loss contributes to drawdown. Fee, rebate, and impact terms remain in the
    reward path and are intentionally excluded from this gross MDD.
    """
    make_policy = policy if policy is not None else _default_policy
    if isinstance(env_factories, str):
        raise TypeError("env_factories must be a factory, (name, factory), or list of those")

    if callable(env_factories):
        pairs: list[tuple[str, Callable[[], OrderBookEnv]]] = [("default", env_factories)]
    elif isinstance(env_factories, tuple) and len(env_factories) == 2:
        pairs = [env_factories]  # type: ignore[list-item]
    else:
        pairs = list(env_factories)  # type: ignore[arg-type]

    rows: list[dict] = []
    for regime_label, factory in pairs:
        for ep in range(n_episodes):
            env = factory()
            env.reset(seed=seed + ep)
            done = False
            mtm_path = [float(env.mark_to_market())]
            reward_accum = 0.0
            while not done:
                a = make_policy(env)
                _, step_reward, terminated, truncated, info = env.step(a)
                reward_accum += float(step_reward)
                mtm_path.append(float(info.get("pnl_ticks", 0.0)))
                done = terminated or truncated
            leftover = int(info.get("inventory", env.inventory))
            fills = env.fills
            arrival = env.arrival_mid
            market_vwap = float(info.get("market_vwap", 0.0) or 0.0)
            rows.append(
                {
                    "regime": regime_label,
                    "policy": name,
                    "reward": float(reward_accum),
                    "is_bps": implementation_shortfall(fills, arrival, side=1),
                    "vwap_slip_bps": vwap_slippage(fills, market_vwap, side=1),
                    "fill_rate": fill_rate(fills, env.inventory0),
                    "completion_rate": completion_rate(fills, env.inventory0, leftover),
                    "leftover": leftover,
                    "mdd_ticks": max_drawdown(mtm_path),
                    "steps": env.t,
                }
            )
    return rows


def summarize(rows: Sequence[dict]) -> list[dict]:
    """Collapse per-episode rows into per-(regime, policy) means + CI-naive std."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["regime"], r["policy"]), []).append(r)
    out: list[dict] = []
    for (regime, policy), eps in groups.items():
        n = len(eps)
        out.append(
            {
                "regime": regime,
                "policy": policy,
                "n": n,
                "reward": sum(e["reward"] for e in eps) / n,
                "is_bps": sum(e["is_bps"] for e in eps) / n,
                "vwap_slip_bps": sum(e["vwap_slip_bps"] for e in eps) / n,
                "fill_rate": sum(e["fill_rate"] for e in eps) / n,
                "completion_rate": sum(e["completion_rate"] for e in eps) / n,
                "leftover": sum(e["leftover"] for e in eps) / n,
                "mdd_ticks": sum(e["mdd_ticks"] for e in eps) / n,
                "steps": sum(e["steps"] for e in eps) / n,
            }
        )
    return out
