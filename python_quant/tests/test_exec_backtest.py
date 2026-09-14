"""Execution metrics + backtest tests (plan_2.md Phase 2).

Covers the honesty fixes this layer exists for:
  * slippage benchmarks vs MARKET VWAP, never the strategy's own VWAP;
  * positive-slippage = a cost (same sign as env ``shortfall_bps``);
  * cost money terms fold into the env reward, NOT into the pre-cost
    timing-skill slippage columns the strategies are compared on.
Plus MDD on a hand-built path and the run_backtest/summarize harness.
"""
from __future__ import annotations

import pytest
from nexus_quant.envs.order_book_env import OrderBookEnv
from nexus_quant.execution.backtest import run_backtest, summarize
from nexus_quant.execution.metrics import (
    completion_rate,
    execution_vwap,
    fill_rate,
    implementation_shortfall,
    inv_risk,
    max_drawdown,
    vwap_slippage,
)


def test_max_drawdown_hand_built_path() -> None:
    path = [0.0, 10.0, 5.0, 20.0, 3.0, 15.0]
    # peak runs 0→10→20; trough 20→3 ⇒ MDD 17
    assert max_drawdown(path) == pytest.approx(17.0)
    # The reset mark is part of the path: a first-step loss is drawdown even
    # when every later mark recovers above zero.
    assert max_drawdown([0.0, -7.0, 4.0]) == pytest.approx(7.0)
    # A recovery after the trough does not erase the peak-to-trough loss.
    assert max_drawdown([0.0, 10.0, 3.0, 11.0]) == pytest.approx(7.0)
    assert max_drawdown([]) == 0.0
    assert max_drawdown([1.0, 2.0, 3.0]) == 0.0  # monotonic up ⇒ no drawdown
    assert max_drawdown([7.0]) == 0.0


def test_vwap_slippage_benchmarks_market_not_self() -> None:
    """Slicker fill than the tape must read NEGATIVE (good); the benchmark is
    the tape's market VWAP even when the strategy's own VWAP is a single price."""
    fills_above = [(15000, 100), (15000, 150)]  # self_vwap = 15000 (sold above 14990 tape)
    assert execution_vwap(fills_above) == pytest.approx(15000.0)
    slip = vwap_slippage(fills_above, 14990.0, side=1)
    assert slip == pytest.approx((14990.0 - 15000.0) / 14990.0 * 1e4)
    assert slip < 0.0  # sold above the tape → favourable, negative cost

    fills_below = [(14980, 100)]  # self_vwap 14980 < tape 15000 → sold cheap
    assert vwap_slippage(fills_below, 15000.0, side=1) > 0.0  # cost

    # no market vwap / no fills → zero, not an error
    assert vwap_slippage(fills_above, 0.0, side=1) == 0.0
    assert vwap_slippage([], 15000.0, side=1) == 0.0


def test_implementation_shortfall_signs() -> None:
    # sell below arrival → positive cost; buy above arrival → positive cost
    assert implementation_shortfall([(14995, 100)], 15000.0, side=1) == pytest.approx(
        (15000.0 - 14995.0) / 15000.0 * 1e4
    )
    assert implementation_shortfall([(15005, 100)], 15000.0, side=-1) == pytest.approx(
        (-1.0) * (15000.0 - 15005.0) / 15000.0 * 1e4
    )
    # at-arrival execution → zero; no fills → zero
    assert implementation_shortfall([(15000, 100)], 15000.0, side=1) == 0.0
    assert implementation_shortfall([], 15000.0, side=1) == 0.0


def test_is_matches_env_shortfall_bps() -> None:
    """The metrics-layer IS is exactly the env's own shortfall_bps (sell side)."""
    env = OrderBookEnv(seed=9)
    env.reset(seed=9)
    done = False
    info: dict = {}
    while not done:
        _, _, term, trunc, info = env.step(0.0)
        done = term or trunc
    assert implementation_shortfall(env.fills, env.arrival_mid, side=1) == pytest.approx(
        info["shortfall_bps"]
    )


def test_fill_and_completion_rates() -> None:
    fills = [(15000, 100), (14999, 150)]  # 250 shares filled
    assert fill_rate(fills, 250) == pytest.approx(1.0)
    assert fill_rate(fills, 500) == pytest.approx(0.5)
    assert fill_rate(fills, 0) == 0.0
    assert completion_rate(fills, 250, 0) == pytest.approx(1.0)
    assert completion_rate(fills, 250, 100) == pytest.approx(250 / 350)


def test_inv_risk_abs_units_and_zero_edge() -> None:
    assert inv_risk(0.2, -500, 0.25) == pytest.approx(0.2 * 500 * 0.5)
    assert inv_risk(0.2, 500, 0.0) == 0.0
    assert inv_risk(0.2, 0, 0.25) == 0.0


def test_execution_vwap_shapes_and_empty() -> None:
    assert execution_vwap([]) == 0.0
    assert execution_vwap([(1, 15000, 100), (2, 15010, 100)]) == pytest.approx(15005.0)
    assert execution_vwap([(15000, 100)]) == pytest.approx(15000.0)  # (px, sz) shape


def _env() -> OrderBookEnv:
    return OrderBookEnv(seed=3)


class _PartialFirstLossEnv:
    """One deterministic truncated episode for backtest metric pins."""

    inventory0 = 100
    arrival_mid = 100

    def reset(self, *, seed: int | None = None):
        del seed
        self.inventory = self.inventory0
        self.fills: list[tuple[int, int, int]] = []
        self.t = 0
        return None, {"arrival_mid": self.arrival_mid}

    def mark_to_market(self) -> float:
        return 0.0

    def step(self, action: float):
        del action
        self.t = 1
        self.inventory = 60
        self.fills = [(1, 95, 40)]
        return None, -1.0, False, True, {
            "inventory": self.inventory,
            "market_vwap": 100.0,
            "pnl_ticks": -500.0,
        }


def test_run_backtest_rows_and_regime_shapes() -> None:
    rows = run_backtest(_env, n_episodes=3, seed=100)
    assert len(rows) == 3
    keys = {
        "regime", "policy", "reward", "is_bps", "vwap_slip_bps", "fill_rate",
        "completion_rate", "leftover", "mdd_ticks", "steps",
    }
    assert keys <= set(rows[0])
    assert all(r["regime"] == "default" and r["policy"] == "policy" for r in rows)

    rows_tuple = run_backtest(("calm", _env), n_episodes=2, seed=101)
    assert len(rows_tuple) == 2
    assert all(r["regime"] == "calm" for r in rows_tuple)

    rows_multi = run_backtest([("a", _env), ("b", _env)], n_episodes=2, seed=102)
    assert [r["regime"] for r in rows_multi] == ["a", "a", "b", "b"]

    with pytest.raises(TypeError):
        run_backtest("not-a-factory")  # type: ignore[arg-type]


def test_run_backtest_partial_fill_counts_first_step_loss_from_reset() -> None:
    rows = run_backtest(_PartialFirstLossEnv, n_episodes=1, seed=1)
    assert rows[0]["fill_rate"] == pytest.approx(0.4)
    assert rows[0]["completion_rate"] == pytest.approx(0.4)
    # The path is [reset=0, first-step=-500], not just [-500].
    assert rows[0]["mdd_ticks"] == pytest.approx(500.0)


def test_run_backtest_custom_policy_and_summarize() -> None:
    market = lambda env: -1.0
    rows_a = run_backtest(("mkt", _env), policy=market, n_episodes=3, seed=7)
    rows_b = run_backtest(("mkt", _env), policy=market, n_episodes=3, seed=7)
    assert rows_a == rows_b  # deterministic under the same seed

    summ = summarize(rows_a + rows_b)  # 6 rows, 2 (regime, policy) groups
    assert len(summ) == 1
    assert summ[0]["n"] == 6
    assert all(k in summ[0] for k in ("is_bps", "vwap_slip_bps", "fill_rate", "mdd_ticks", "steps"))
    assert summ[0]["mdd_ticks"] == pytest.approx(sum(r["mdd_ticks"] for r in rows_a + rows_b) / 6)


def test_run_backtest_cost_folds_into_reward_not_slippage() -> None:
    """Under identical market actions, a taker fee leaves the pre-cost slippage
    columns identical and only moves reward — the fairness envelope for E7."""

    def run(fee_bps: float) -> list[dict]:
        return run_backtest(
            lambda: OrderBookEnv(seed=6, fee_bps=fee_bps),
            policy=lambda env: -1.0,
            n_episodes=3,
            seed=50,
        )

    rows_clean, rows_fee = run(0.0), run(10.0)
    assert len(rows_clean) == len(rows_fee)
    for b, f in zip(rows_clean, rows_fee):
        assert b["is_bps"] == pytest.approx(f["is_bps"])  # pre-cost timing skill unchanged
        assert b["vwap_slip_bps"] == pytest.approx(f["vwap_slip_bps"])
        assert f["reward"] <= b["reward"] + 1e-9  # fee never helps
    assert any(f["reward"] < b["reward"] for b, f in zip(rows_clean, rows_fee))
