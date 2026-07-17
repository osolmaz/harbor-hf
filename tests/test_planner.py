from pathlib import Path

from harbor_hf.io import load_experiment
from harbor_hf.models import DeploymentProfile
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
        "sha256:b03e22e8dd8499873093b0611aa936140f6ec98c27db5a9e4b45375f47562bd5"
    )


def test_digest_preserves_legacy_empty_engine_secret_names() -> None:
    spec = load_experiment(EXAMPLE)
    for deployment in spec.matrix.deployments:
        assert isinstance(deployment, DeploymentProfile)
        deployment.engine.secret_names = []

    assert experiment_digest(spec) == (
        "sha256:99a7524ff6b7e91ff6a47ee226db280608f844381b8407f91b47d2fe7115691c"
    )
    payload = spec.model_dump(mode="json", exclude_none=True)
    assert payload["matrix"]["deployments"][0]["engine"]["secret_names"] == []


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


def test_counts_only_selected_tasks_from_a_larger_digest_map() -> None:
    spec = load_experiment(EXAMPLE).model_copy(update={"remote": None})
    selected = spec.model_copy(
        update={
            "benchmark": spec.benchmark.model_copy(
                update={
                    "task_names": ["selected-*"],
                    "task_digests": {
                        "selected-one": "sha256:" + "1" * 64,
                        "unselected-one": "sha256:" + "2" * 64,
                    },
                }
            )
        }
    )

    assert build_plan(selected).logical_trial_count == 2


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
