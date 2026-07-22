from collections.abc import Collection
from fnmatch import fnmatch


def task_matches_selector(
    task: str, selector: str, available_tasks: Collection[str]
) -> bool:
    """Match a literal task name before interpreting a selector as a pattern."""
    if selector in available_tasks:
        return task == selector
    return fnmatch(task, selector)
