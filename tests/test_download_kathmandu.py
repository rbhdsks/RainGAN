from pathlib import Path

import pytest

from scripts.download_kathmandu import safe_relative_path, selected_datasets


def test_safe_dataverse_path():
    result = safe_relative_path("dataset/images", "frame_0001.jpg")
    assert result == Path("dataset/images/frame_0001.jpg")


@pytest.mark.parametrize(
    ("directory", "label"),
    [("../outside", "image.jpg"), ("/absolute", "image.jpg"), (None, "../image.jpg")],
)
def test_rejects_path_traversal(directory, label):
    with pytest.raises(ValueError):
        safe_relative_path(directory, label)


def test_all_selects_both_releases():
    assert len(selected_datasets("all")) == 2
