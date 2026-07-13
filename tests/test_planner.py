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
        "sha256:8fa81417ab5315c4b36a8b1ba3ddb3fa952db950d37614796e3faa6dbd54ef63"
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


def test_patterns_have_unresolved_trial_counts() -> None:
    spec = load_experiment(EXAMPLE)

    for task in ("task-*", "task?", "task[12]"):
        patterned = spec.model_copy(
            update={
                "benchmark": spec.benchmark.model_copy(update={"task_names": [task]})
            }
        )

        assert build_plan(patterned).logical_trial_count is None
