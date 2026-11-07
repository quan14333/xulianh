from pathlib import Path

import numpy as np
import pytest

from src.analysis.controlled_roi_audit import contrast, masks_for, run_audit


def test_ring_clips_at_border_and_excludes_neighbor_in_same_scene():
    annotations = (
        ("scene", "R1", "", 0, 0, 2, 2),
        ("scene", "R2", "", 2, 0, 3, 2),
        ("other-scene", "R3", "", 0, 2, 2, 3),
    )
    roi, background = masks_for("scene", (0, 0, 2, 2), 1, (5, 5), annotations)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, :3] = True
    np.testing.assert_array_equal(background, expected)
    assert roi.sum() == 4
    assert not np.any(roi & background)


def test_signed_contrast_casts_uint8_before_subtraction_and_distinguishes_median():
    image = np.array([[0, 0, 255], [200, 200, 200]], dtype=np.uint8)
    roi = np.array([[True, True, True], [False, False, False]])
    background = ~roi
    assert contrast(image, roi, background, "mean") == -115.0
    assert contrast(image, roi, background, "median") == -200.0


@pytest.mark.parametrize(
    "box,pad", [((0, 0, 0, 2), 1), ((-1, 0, 2, 2), 1), ((0, 0, 6, 2), 1), ((0, 0, 2, 2), 0)]
)
def test_invalid_roi_or_ring_is_rejected(box, pad):
    with pytest.raises(ValueError):
        masks_for("scene", box, pad, (5, 5), ())


def test_roi_covering_whole_image_has_no_valid_background():
    with pytest.raises(ValueError, match="background"):
        masks_for("scene", (0, 0, 5, 5), 1, (5, 5), ())


@pytest.mark.parametrize("case", ["overlap", "empty", "shape", "nan", "mode"])
def test_invalid_measurements_fail_instead_of_returning_a_score(case):
    image = np.ones((2, 2))
    roi = np.array([[True, False], [False, False]])
    background = ~roi
    mode = "mean"
    if case == "overlap":
        background[0, 0] = True
    elif case == "empty":
        roi[:] = False
    elif case == "shape":
        image = image[:1]
    elif case == "nan":
        image[0, 0] = np.nan
    else:
        mode = "maximum"
    with pytest.raises(ValueError):
        contrast(image, roi, background, mode)


def test_existing_output_is_never_overwritten(tmp_path):
    marker = tmp_path / "evidence.txt"
    marker.write_text("keep existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_audit(Path("missing-sage"), Path("missing-pcd"), tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep existing evidence"
