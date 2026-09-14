"""The dashboard must plot recorded fills, including terminal liquidation."""
from __future__ import annotations

import json

import pytest
from nexus_quant.baselines import FAIR_BASELINES, policy_action
from nexus_quant.dashboard_execution import (
    MAX_HORIZON,
    MAX_INVENTORY,
    MAX_SEED,
    ExecutionConfig,
    execution_summary,
    run_execution_demo,
)
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.envs.regimes import regime_kwargs


def test_summary_uses_exact_proceeds_and_sell_side_sign():
    summary = execution_summary(
        inventory=10, remaining=4, cash_ticks=593,
        arrival_mid_ticks=100, market_vwap_ticks=99,
    )
    assert summary["filled"] == 6
    assert summary["completion_pct"] == 60
    assert summary["execution_vwap_ticks"] == pytest.approx(593 / 6)
    assert summary["arrival_shortfall_bps"] == pytest.approx((100 - 593 / 6) * 100)
    assert summary["market_slippage_bps"] == pytest.approx((99 - 593 / 6) / 99 * 10_000)
    favorable = execution_summary(
        inventory=10, remaining=0, cash_ticks=1010,
        arrival_mid_ticks=100, market_vwap_ticks=100,
    )
    assert favorable["arrival_shortfall_bps"] == -100


def test_summary_does_not_invent_no_fill_or_no_market_metrics():
    summary = execution_summary(
        inventory=10, remaining=10, cash_ticks=0,
        arrival_mid_ticks=100, market_vwap_ticks=None,
    )
    assert summary["completion_pct"] == 0
    for key in (
        "execution_vwap_ticks", "arrival_shortfall_bps",
        "market_vwap_ticks", "market_slippage_bps",
    ):
        assert summary[key] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("strategy", FAIR_BASELINES)
def test_seed_reproduces_every_fill_and_summary(strategy):
    config = ExecutionConfig(strategy=strategy, seed=42)
    assert run_execution_demo(config) == run_execution_demo(config)
    assert run_execution_demo(config)["fills"] != run_execution_demo(
        ExecutionConfig(strategy=strategy, seed=43)
    )["fills"]


@pytest.mark.parametrize("strategy", FAIR_BASELINES)
def test_telemetry_matches_a_direct_environment_run(strategy):
    config = ExecutionConfig(strategy=strategy, seed=123, horizon=12, inventory=3000)
    result = run_execution_demo(config)
    env = OrderBookEnv(
        inventory=config.inventory, horizon=config.horizon, seed=config.seed,
        **regime_kwargs("calm", costs=True, vol_feature=False),
    )
    try:
        env.reset(seed=config.seed)
        for step in range(1, config.horizon + 1):
            previous_inventory = env.inventory
            _, _, terminated, truncated, _ = env.step(policy_action(strategy, env))
            assert result["history"][step] == {
                "step": step,
                "filled": previous_inventory - env.inventory,
                "remaining": env.inventory,
                "cash_ticks": env.cash_ticks,
            }
            if terminated or truncated:
                break
        assert [(f["env_step"], f["price_ticks"], f["quantity"]) for f in result["fills"]] == env.fills
        summary = result["summary"]
        assert summary["cash_ticks"] == env.cash_ticks
        assert summary["market_vwap_ticks"] == env.market_vwap()
        assert summary["filled"] == config.inventory - env.inventory
        assert summary["execution_vwap_ticks"] == pytest.approx(env.cash_ticks / summary["filled"])
    finally:
        env.close()
    assert result["source"] == "synthetic-execution"
    assert result["vol_feature"] is False
    assert result["queue_model"] == "uniform"
    assert "gross" in result["metric_basis"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("horizon,inventory", [(1, 1), (1, MAX_INVENTORY), (MAX_HORIZON, MAX_INVENTORY)])
def test_episode_bounds_and_inventory_reconcile(horizon, inventory):
    result = run_execution_demo(ExecutionConfig(horizon=horizon, inventory=inventory, seed=0))
    summary = result["summary"]
    assert 1 <= summary["steps"] <= horizon
    assert len(result["history"]) == summary["steps"] + 1
    assert len(result["fills"]) <= horizon + 1
    assert result["history"][0] == {"step": 0, "filled": 0, "remaining": inventory, "cash_ticks": 0}
    previous = inventory
    for row in result["history"]:
        recorded_quantity = sum(f["quantity"] for f in result["fills"] if f["step"] == row["step"])
        assert row["filled"] == recorded_quantity == previous - row["remaining"]
        assert 0 <= row["remaining"] <= previous
        previous = row["remaining"]
    assert summary["remaining"] == previous
    assert sum(f["quantity"] for f in result["fills"]) + previous == inventory
    for fill in result["fills"]:
        assert type(fill["price_ticks"]) is int
        assert type(fill["quantity"]) is int and fill["quantity"] > 0
        assert 1 <= fill["step"] <= summary["steps"]


def test_horizon_includes_real_terminal_fill_without_assuming_completion():
    result = run_execution_demo(ExecutionConfig(horizon=1, inventory=MAX_INVENTORY, seed=0))
    terminal = [f for f in result["fills"] if f["kind"] == "terminal"]
    assert terminal
    assert all(f["step"] == f["env_step"] == 1 for f in terminal)
    assert result["summary"]["terminal_filled"] == sum(f["quantity"] for f in terminal)
    assert result["summary"]["stop_reason"] == "horizon"
    assert result["summary"]["remaining"] > 0
    assert result["summary"]["completion_pct"] < 100


@pytest.mark.parametrize("payload", [
    None, [], "schedule_twap", {"unknown": 1}, {"strategy": "ppo"},
    {"strategy": "twap"}, {"strategy": []}, {"seed": -1}, {"seed": MAX_SEED + 1},
    {"seed": True}, {"seed": 1.5}, {"seed": "1"}, {"seed": None},
    {"horizon": 0}, {"horizon": MAX_HORIZON + 1}, {"horizon": 1.0},
    {"inventory": 0}, {"inventory": MAX_INVENTORY + 1}, {"inventory": False},
])
def test_invalid_demo_parameters_are_rejected(payload):
    with pytest.raises((TypeError, ValueError)):
        ExecutionConfig.from_payload(payload)


def test_default_and_largest_seed_are_accepted():
    assert ExecutionConfig.from_payload({}) == ExecutionConfig()
    assert run_execution_demo(ExecutionConfig(seed=MAX_SEED, horizon=1))["config"]["seed"] == MAX_SEED
