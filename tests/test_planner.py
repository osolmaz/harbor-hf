from pathlib import Path

from harbor_hf.io import load_experiment
from harbor_hf.planner import build_plan, experiment_digest

EXAMPLE = Path(__file__).parent.parent / "examples" / "shellbench.yaml"


def test_builds_cartesian_plan() -> None:
    spec = load_experiment(EXAMPLE)
    plan = build_plan(spec)

    assert plan.run_count == 2
    assert plan.logical_trial_count == 2
    assert [(cell.model, cell.deployment, cell.agent) for cell in plan.cells] == [
        ("qwen36-nvfp4", "h200", "openclaw"),
        ("qwen36-nvfp4", "rtx-pro-6000", "openclaw"),
    ]


def test_digest_is_stable() -> None:
    spec = load_experiment(EXAMPLE)

    assert experiment_digest(spec) == experiment_digest(spec.model_copy(deep=True))
    assert experiment_digest(spec) == (
        "sha256:af4e4c36c687d31e44dad588dee0bc8397939e02b62ee316be7663847702f233"
    )


def test_counts_explicit_tasks_and_attempts() -> None:
    spec = load_experiment(EXAMPLE)
    explicit = spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(
                update={
                    "task_names": ["task-one", "task-two"],
                    "task_digests": {
                        "task-one": "sha256:" + "1" * 64,
                        "task-two": "sha256:" + "2" * 64,
                    },
                }
            ),
            "execution": spec.execution.model_copy(update={"attempts": 3}),
        }
    )

    plan = build_plan(explicit)

    assert plan.run_count == 2
    assert plan.logical_trial_count == 12


def test_patterns_have_unresolved_trial_counts() -> None:
    spec = load_experiment(EXAMPLE).model_copy(update={"remote": None})

    for task in ("task-*", "task?", "task[12]"):
        patterned = spec.model_copy(
            update={
                "benchmark": spec.benchmark.model_copy(
                    update={"task_names": [task], "task_digests": {}}
                )
            }
        )

        assert build_plan(patterned).logical_trial_count is None
