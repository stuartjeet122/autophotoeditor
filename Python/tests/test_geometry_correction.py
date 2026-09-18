import numpy as np

from autophotoeditor.processing import geometry_correction


def _estimate() -> geometry_correction.GeometryEstimate:
    return geometry_correction.GeometryEstimate(
        level=2.0,
        vertical_strength=20.0,
        horizontal_strength=-30.0,
        vertical_confidence=0.9,
        horizontal_confidence=0.9,
        vertical_vp=np.array([600.0, 400.0]),
        horizontal_vp=np.array([500.0, 200.0]),
    )


def test_each_geometry_mode_applies_its_expected_correction(monkeypatch) -> None:
    image = np.zeros((800, 1000, 3), dtype=np.uint8)
    monkeypatch.setattr(
        geometry_correction,
        "analyze_geometry",
        lambda image, max_lines, verbose: _estimate(),
    )

    transforms = {
        mode: geometry_correction.build_transform(
            image,
            mode=mode,
            verbose=False,
        )[0]
        for mode in ("auto", "level", "vertical", "horizontal", "full")
    }

    assert not np.allclose(transforms["auto"], np.eye(3))
    assert not np.allclose(transforms["level"], np.eye(3))
    assert not np.allclose(transforms["vertical"], transforms["level"])
    assert not np.allclose(transforms["horizontal"], transforms["level"])
    assert not np.allclose(transforms["full"], transforms["vertical"])
    assert not np.allclose(transforms["full"], transforms["horizontal"])


def test_projective_strength_preserves_signed_vp_direction() -> None:
    positive = geometry_correction.vp_projective_transform(
        20.0,
        0.0,
        1000,
        800,
    )
    negative = geometry_correction.vp_projective_transform(
        -20.0,
        0.0,
        1000,
        800,
    )

    assert positive[2, 0] < 0.0
    assert negative[2, 0] > 0.0
    assert np.isclose(abs(positive[2, 0]), abs(negative[2, 0]))


def test_full_correction_maps_vps_to_axis_infinity() -> None:
    vertical_vp = np.array([600.0, 520.0])
    horizontal_vp = np.array([420.0, 200.0])
    transform = geometry_correction.vp_projective_transform(
        20.0,
        -30.0,
        1000,
        800,
        vertical_vp=vertical_vp,
        horizontal_vp=horizontal_vp,
        vertical_reference=20.0,
        horizontal_reference=-30.0,
    )

    vertical_mapped = transform @ np.r_[vertical_vp, 1.0]
    horizontal_mapped = transform @ np.r_[horizontal_vp, 1.0]

    assert np.isclose(vertical_mapped[2], 0.0)
    assert np.isclose(horizontal_mapped[2], 0.0)
    assert np.isclose(vertical_mapped[0], 0.0)
    assert np.isclose(horizontal_mapped[1], 0.0)