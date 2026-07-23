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
        "sha256:e4e041c0a281516917ac2180fee84e2ce23feed730e459c28a9e0c6d1bff4199"
    )


def test_digest_preserves_legacy_empty_engine_secret_names() -> None:
    spec = load_experiment(EXAMPLE)
    for deployment in spec.matrix.deployments:
        assert isinstance(deployment, DeploymentProfile)
        deployment.engine.secret_names = []

    assert experiment_digest(spec) == (
        "sha256:f9b4baf6345b9abe06087922c50c59c53a6e935c1aee9048a9530a457aff435b"
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
