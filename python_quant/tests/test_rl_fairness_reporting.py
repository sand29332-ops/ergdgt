"""E7 reporting: operational metrics, cache provenance, and no reruns."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.agents import PPOPolicy
from nexus_quant.agents import evaluate as evaluate_module
from scripts import rl_fairness_study as study


def test_policy_cache_requires_exact_training_provenance(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_train(_factory, cfg):
        calls.append(cfg)
        return PPOPolicy(obs_dim=44, hidden=cfg.hidden, seed=cfg.seed), []

    monkeypatch.setattr(study, "train_ppo", fake_train)
    study._train("novol", 0, 2, 2, tmp_path)
    study._train("novol", 0, 2, 2, tmp_path)

    assert len(calls) == 1  # exact config reuses
    _, first = study._training_spec("novol", 0, 2, 2)
    _, changed = study._training_spec("novol", 0, 3, 2)
    first_policy = study._cache_path(tmp_path, first)
    changed_policy = study._cache_path(tmp_path, changed)
    np.savez(first_policy, **{study._CACHE_PROVENANCE_FIELD: np.asarray("{}")})
    study._train("novol", 0, 2, 2, tmp_path)
    study._train("novol", 0, 3, 2, tmp_path)

    assert len(calls) == 3  # stale embedded metadata and changed iterations both retrain
    assert first["cache_key"] != changed["cache_key"]
    assert first_policy.is_file() and changed_policy.is_file()
    with np.load(first_policy, allow_pickle=False) as saved:
        assert json.loads(str(saved[study._CACHE_PROVENANCE_FIELD].item())) == first


def _fake_episode_rows(
    policy,
    regimes,
    *,
    seeds: int,
    episodes_per_seed: int,
    seed0: int,
    baselines=(),
    agent_name: str = "ppo",
    **_kwargs,
):
    names = ([agent_name] if policy is not None else []) + list(baselines)
    out = {}
    for regime in regimes:
        out[regime] = {}
        for index, name in enumerate(names):
            is_ppo = name == agent_name and policy is not None
            rows = []
            for family in range(seeds):
                for episode in range(episodes_per_seed):
                    rows.append(
                        {
                            "seed": seed0 + family * 10_000 + episode,
                            "seed_family": family,
                            "reward": 2.0 if is_ppo else 1.0 - index,
                            "shortfall_bps": 1.0 if is_ppo else 2.0 + index,
                            "is_bps": 1.0 if is_ppo else 2.0 + index,
                            "vwap_slip_bps": 0.5 if is_ppo else 1.0 + index,
                            "leftover": 60 if is_ppo else 20,
                            "completion": 0.4 if is_ppo else 0.8,
                            "fill_rate": 0.4 if is_ppo else 0.8,
                            "mdd_ticks": 50.0 if is_ppo else 25.0 + index,
                        }
                    )
            out[regime][name] = rows
    return out


def test_e7_report_uses_existing_rollouts_and_renders_operational_cis(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_train(*_args):
        return object()

    def counting_rows(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_episode_rows(*args, **kwargs)

    def unexpected_rollout(*_args, **_kwargs):
        pytest.fail("paired CIs must consume the already-collected E7 rows")

    monkeypatch.setattr(study, "_train", fake_train)
    monkeypatch.setattr(study, "run_regime_episodes", counting_rows)
    monkeypatch.setattr(evaluate_module, "run_regime_episodes", unexpected_rollout)
    result = study.run(
        argparse.Namespace(
            iters=2,
            episodes=2,
            train_seeds=2,
            eval_seeds=1,
            episodes_per_seed=2,
            n_boot=20,
            artifacts=str(tmp_path),
        )
    )

    # Once per mode for baselines, then once per trained PPO; paired CIs only
    # consume those rows and never launch an additional episode rollout.
    assert len(calls) == len(study.MODES) * 3
    assert sum(call[0][0] is None for call in calls) == len(study.MODES)
    assert all(call[1]["seed0"] == study.EVAL_SEED0 for call in calls)

    assert result["config"]["report_schema_version"] == study.REPORT_SCHEMA_VERSION
    assert {"fill_rate", "mdd_ticks"} <= set(result["config"]["metrics"])
    assert "price-tick × share" in result["config"]["metric_units"]["mdd_ticks"]
    for mode in study.MODES:
        mode_result = result["modes"][mode]
        assert len(mode_result["training_provenance"]) == 2
        assert mode_result["training_provenance"][0]["ppo_config"]["iterations"] == 2
        for metric in ("fill_rate", "mdd_ticks"):
            values = mode_result["per_metric"][metric]["highvol"]
            assert {"mean", "lo", "hi", "n", "per_seed_mean"} <= set(values["ppo_pooled_ci"])
            assert {"mean", "lo", "hi"} <= set(values["baselines"]["twap"])

    rendered = study.render_markdown(result)
    assert "Fill rate (filled shares / parent shares) (mean [95% CI]):" in rendered
    assert "Gross mark-to-market MDD (price-tick × shares) (mean [95% CI]):" in rendered
    assert "| ppo |" in rendered and "| twap |" in rendered

    legacy = copy.deepcopy(result)
    for mode in study.MODES:
        legacy_metrics = legacy["modes"][mode]["per_metric"]
        legacy_metrics.pop("fill_rate")
        legacy_metrics.pop("mdd_ticks")
    assert "not reported in this legacy result" in study.render_markdown(legacy)


@pytest.mark.parametrize(
    "argv",
    (
        ("--iters", "0"),
        ("--episodes", "-1"),
        ("--train-seeds", "0"),
        ("--eval-seeds", "0"),
        ("--episodes-per-seed", "0"),
        ("--n-boot", "0"),
        ("--eval-seeds", "1", "--episodes-per-seed", "1"),
    ),
)
def test_fairness_cli_rejects_invalid_ci_counts(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        study.main(list(argv))
    assert excinfo.value.code == 2
