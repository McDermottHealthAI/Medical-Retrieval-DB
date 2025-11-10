"""Test set-up and fixtures code."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from yaml_to_disk import yaml_disk


def get_test_data(test_data: str | None = None):
    """Create a temporary directory with medical documents for testing.

    Args:
        test_data: Optional YAML string defining test documents. If None, uses default
                   sample medical documents (diabetes.txt and heart.txt).

    Returns:
        A context manager that yields a temporary directory containing test medical documents.

    Examples:
        Using default test data:

        >>> with get_test_data() as temp_dir:
        ...     files = list(temp_dir.iterdir())
        ...     len(files)
        2
        >>> files[0].name
        'diabetes.txt'

        Using custom test data:

        >>> custom_data = '''
        ... disease.txt: "A new disease description."
        ... '''
        >>> with get_test_data(custom_data) as temp_dir:
        ...     files = list(temp_dir.iterdir())
        ...     len(files)
        1
    """
    if test_data is None:
        test_data = """
            diabetes.txt: "Diabetes is a chronic condition that affects blood sugar levels."
            heart.txt: "Heart disease is a leading cause of death worldwide."
        """
    return yaml_disk(test_data)


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    # You can pass more fixtures here to add them to the namespace
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "get_test_data": get_test_data,
        }
    )
