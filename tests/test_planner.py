from pathlib import Path

from harbor_hf.io import load_experiment
from harbor_hf.planner import build_plan, experiment_digest

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


def test_builds_cartesian_plan() -> None:
    spec = load_experiment(EXAMPLE)
    plan = build_plan(spec)

    assert plan.run_count == 2
    assert plan.logical_trial_count is None
    assert [(cell.model, cell.deployment, cell.agent) for cell in plan.cells] == [
        ("qwen36-nvfp4", "rtx-pro-6000", "openclaw"),
        ("qwen36-nvfp4", "h200", "openclaw"),
    ]


def test_digest_is_stable() -> None:
    spec = load_experiment(EXAMPLE)

    assert experiment_digest(spec) == experiment_digest(spec.model_copy(deep=True))
    assert experiment_digest(spec) == (
        "sha256:2bff6c87bd06b1fed18b6e0249150dcef0c4d7ea3aa6658284727ce0b944e5a6"
    )


def test_counts_explicit_tasks_and_attempts() -> None:
    spec = load_experiment(EXAMPLE)
    explicit = spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(
                update={"task_names": ["task-one", "task-two"]}
            ),
            "execution": spec.execution.model_copy(update={"attempts": 3}),
        }
    )

    plan = build_plan(explicit)

    assert plan.run_count == 2
    assert plan.logical_trial_count == 12
