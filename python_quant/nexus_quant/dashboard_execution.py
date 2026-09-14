"""Bounded, request-scoped execution telemetry, separate from the L2 feed."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .baselines import FAIR_BASELINES, policy_action
from .envs.order_book_env import OrderBookEnv
from .envs.regimes import regime_kwargs

EXECUTION_SOURCE = "synthetic-execution"
MAX_HORIZON = 80
MAX_INVENTORY = 10_000
MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class ExecutionConfig:
    strategy: str = "schedule_twap"
    seed: int = 0x51ED
    horizon: int = 40
    inventory: int = 2_000

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or self.strategy not in FAIR_BASELINES:
            raise ValueError("strategy must be schedule_twap, adaptive_pov or is_aware")
        for name, low, high in (
            ("seed", 0, MAX_SEED),
            ("horizon", 1, MAX_HORIZON),
            ("inventory", 1, MAX_INVENTORY),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionConfig:
        if not isinstance(payload, dict):
            raise TypeError("body must be a JSON object")
        if payload.keys() - {"strategy", "seed", "horizon", "inventory"}:
            raise ValueError("unknown execution parameter")
        return cls(**payload)


def execution_summary(
    *,
    inventory: int,
    remaining: int,
    cash_ticks: int,
    arrival_mid_ticks: float,
    market_vwap_ticks: float | None,
) -> dict[str, Any]:
    """Gross, filled-quantity metrics; never round cash via aggregate fill prices."""
    filled = inventory - remaining
    vwap = cash_ticks / filled if filled else None
    arrival_is = (
        (arrival_mid_ticks - vwap) / arrival_mid_ticks * 1e4
        if vwap is not None and arrival_mid_ticks > 0 else None
    )
    market_slip = (
        (market_vwap_ticks - vwap) / market_vwap_ticks * 1e4
        if vwap is not None and market_vwap_ticks and market_vwap_ticks > 0 else None
    )
    return {
        "filled": filled,
        "remaining": remaining,
        "completion_pct": filled / inventory * 100,
        "cash_ticks": cash_ticks,
        "arrival_mid_ticks": arrival_mid_ticks,
        "execution_vwap_ticks": vwap,
        "market_vwap_ticks": market_vwap_ticks or None,
        "arrival_shortfall_bps": arrival_is,
        "market_slippage_bps": market_slip,
    }


def run_execution_demo(config: ExecutionConfig) -> dict[str, Any]:
    """Run a fresh calm-regime sell episode; do not mutate the shared book hub."""
    env = OrderBookEnv(
        inventory=config.inventory,
        horizon=config.horizon,
        seed=config.seed,
        **regime_kwargs("calm", costs=True, vol_feature=False),
    )
    env.reset(seed=config.seed)
    fills: list[dict[str, Any]] = []
    history = [{"step": 0, "filled": 0, "remaining": config.inventory, "cash_ticks": 0}]
    stopped = "horizon"
    try:
        for step in range(1, config.horizon + 1):
            previous_count = len(env.fills)
            previous_inventory = env.inventory
            action = policy_action(config.strategy, env)
            _, _, terminated, truncated, info = env.step(action)
            new_fills = env.fills[previous_count:]
            filled = sum(int(quantity) for _, _, quantity in new_fills)
            if (
                env.t != step or env.inventory < 0
                or filled != previous_inventory - env.inventory
            ):
                raise RuntimeError("execution inventory did not reconcile with recorded fills")
            for fill_step, price, quantity in new_fills:
                fills.append({
                    "step": step,
                    "env_step": int(fill_step),
                    "price_ticks": int(price),
                    "quantity": int(quantity),
                    "kind": "terminal" if fill_step == env.t else info["mode"],
                })
            history.append({
                "step": step,
                "filled": filled,
                "remaining": int(env.inventory),
                "cash_ticks": int(env.cash_ticks),
            })
            if terminated or truncated:
                stopped = "horizon" if truncated else "filled"
                break
        else:
            raise RuntimeError("execution did not stop within its horizon")

        summary = execution_summary(
            inventory=config.inventory,
            remaining=int(env.inventory),
            cash_ticks=int(env.cash_ticks),
            arrival_mid_ticks=float(env.arrival_mid),
            market_vwap_ticks=float(env.market_vwap()) or None,
        )
        summary.update({
            "steps": int(env.t),
            "stop_reason": stopped,
            "terminal_filled": sum(f["quantity"] for f in fills if f["kind"] == "terminal"),
        })
        return {
            "ok": True,
            "source": EXECUTION_SOURCE,
            "config": asdict(config),
            "regime": "calm",
            "vol_feature": False,
            "queue_model": env.queue_model,
            "metric_basis": "gross fills; fees, rebates and impact affect reward, not cash",
            "fill_price_basis": "environment aggregate integer tick prices, floored for level sweeps",
            "fills": fills,
            "history": history,
            "summary": summary,
        }
    finally:
        env.close()
