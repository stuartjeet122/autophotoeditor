import base64

import cv2
import numpy as np

from autophotoeditor.api.app import masked_enhance_json_api


def test_masked_enhance_json_does_not_return_mask_dir(monkeypatch, tmp_path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    output_path = tmp_path / "enhanced.png"
    output_path.write_bytes(b"fake-png-data")

    monkeypatch.setattr(
        "autophotoeditor.api.app.masked_enhance",
        lambda *args, **kwargs: {
            "output": str(output_path),
            "masks_json": "/tmp/masks.json",
        },
    )

    result = masked_enhance_json_api(
        payload={"image_base64": image_base64, "mask_json": {"masks": {}}},
        masks=None,
        strength=1.0,
        feather=2.0,
        job_id="test-job",
    )

    assert "mask_dir" not in result
    assert result["masks_json"] == "/tmp/masks.json"
    assert result["output_media_type"] == "image/png"
