import base64
import json

from autophotoeditor.api.app import _mask_json_artifact


def test_mask_json_artifact_embeds_declared_binary_file(tmp_path) -> None:
	mask_json = tmp_path / "photo_masks.json"
	binary_dir = tmp_path / "binary_masks"
	binary_dir.mkdir()

	binary_payload = b"mask-png"
	(binary_dir / "photo_mask.png").write_bytes(binary_payload)
	mask_json.write_text(
		json.dumps(
			{
				"masks": {
					"subject": {
						"binary_file": "binary_masks/photo_mask.png",
					}
				}
			}
		),
		encoding="utf-8",
	)

	result = _mask_json_artifact(mask_json, binary_dir)

	subject = result["masks"]["subject"]
	assert subject["binary_encoding"] == "base64"
	assert subject["binary_png_base64"] == base64.b64encode(binary_payload).decode("ascii")
