import pandas as pd
import pytest

from marderlab_tools.metadata.merge import merge_metadata_tabs, normalize_columns


def test_normalize_columns_uses_aliases() -> None:
    frame = pd.DataFrame({"NotebookPage": ["997_001"], "file_number": [1], "stim": [2]})
    normalized = normalize_columns(
        frame,
        {
            "notebook_page": ["NotebookPage"],
            "file_index": ["file_number"],
            "stim_index": ["stim"],
        },
    )
    assert {"notebook_page", "file_index", "stim_index"}.issubset(normalized.columns)


def test_merge_metadata_tabs_requires_fields() -> None:
    frame = pd.DataFrame(
        {
            "notebook_page": ["997_001"],
            "file_index": [1],
            "stim_index": [0],
            "temperature": [12.0],
            "condition": ["control"],
            "experiment_type": ["contracture"],
            "season": [20],
        }
    )
    merged = merge_metadata_tabs(
        {"FTBath": frame},
        column_map={},
        required_fields=[
            "notebook_page",
            "file_index",
            "stim_index",
            "temperature",
            "condition",
            "experiment_type",
            "season",
        ],
    )
    assert len(merged) == 1

    with pytest.raises(ValueError):
        merge_metadata_tabs(
            {"FTBath": frame.drop(columns=["condition"])},
            column_map={},
            required_fields=["notebook_page", "file_index", "condition"],
        )
