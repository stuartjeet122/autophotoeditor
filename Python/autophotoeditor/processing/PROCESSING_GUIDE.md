# Processing Guide

This folder contains the image-processing engines used by the AutoPhotoEditor API.

## How The Packages Work

- `engine.py` is a compatibility facade. Existing imports can continue using `autophotoeditor.processing.<engine>`.
- Action modules contain the implementation for one responsibility.
- `__init__.py` exposes the engine facade and, where useful, the action modules.
- The main API callers are in `autophotoeditor/api/app.py`.
- The normal runtime entry point for command-line modules is each package's `cli.py`.

## AI Denoise

### `ai_denoise/engine.py`
- **Description:** Original implementation and compatibility source for AI denoising.
- **Use it when:** Supporting legacy imports or tracing the complete denoising flow.
- **Used by:** `ai_denoise/__init__.py`, `ai_denoise/download.py`, `ai_denoise/inference.py`, `ai_denoise/fallback.py`, `ai_denoise/cli.py`, and `autophotoeditor/api/app.py`.
- **Functions:** `download_model`, `ai_denoise`, `traditional_fallback`, `main`.

### `ai_denoise/download.py`
- **Description:** Downloads and locates the SwinIR ONNX model.
- **Use it when:** Preparing the model before AI inference.
- **Used by:** `ai_denoise/engine.py` and the denoising CLI flow.
- **Functions:** `download_model`.

### `ai_denoise/inference.py`
- **Description:** Runs tiled ONNX denoising with CPU/CUDA provider selection.
- **Use it when:** Applying AI denoising to an image.
- **Used by:** `ai_denoise/engine.py` and `autophotoeditor/api/app.py`.
- **Functions:** `ai_denoise`, `AIDenoiseError`.

### `ai_denoise/fallback.py`
- **Description:** Provides OpenCV LAB-space denoising when AI inference is unavailable.
- **Use it when:** ONNX Runtime or model inference fails.
- **Used by:** `ai_denoise/engine.py`.
- **Functions:** `traditional_fallback`.

### `ai_denoise/cli.py`
- **Description:** Command-line entry point for denoising an input file.
- **Use it when:** Running denoising from a terminal.
- **Used by:** `python -m autophotoeditor.processing.ai_denoise.cli` or a direct launcher.
- **Functions:** `main`.

### `ai_denoise/__init__.py`
- **Description:** Public package exports and action-module discovery.
- **Use it when:** Importing the denoising engine.
- **Used by:** API code and callers importing `autophotoeditor.processing.ai_denoise`.
- **Functions:** Re-exports engine functions; no implementation functions.

## Preset Application

### `apply_preset/engine.py`
- **Description:** Compatibility facade for the preset processor.
- **Use it when:** Preserving old imports such as `autophotoeditor.processing.apply_preset`.
- **Used by:** `apply_preset/__init__.py` and `autophotoeditor/api/app.py`.
- **Functions:** Re-exports `process_image`, parsing, adjustment, and CLI functions.

### `apply_preset/core.py`
- **Description:** Preset data classes and numeric helper functions.
- **Use it when:** Defining XMP settings or shared numeric operations.
- **Used by:** `parsing.py`, `color.py`, `io.py`, `adjustments.py`, `cli.py`, and `pipeline.py`.
- **Functions/classes:** `HSLAdjustment`, `ToneSettings`, `ColorSettings`, `WhiteBalanceSettings`, `EffectsSettings`, `SharpenSettings`, `NoiseSettings`, `PointCurve`, `XMPSettings`, `PipelineOptions`, `clamp01`, `safe_float`, `finite_image`, `smoothstep`.

### `apply_preset/parsing.py`
- **Description:** Reads Lightroom/XMP preset XML and point curves.
- **Use it when:** Converting an XMP preset into `XMPSettings`.
- **Used by:** `pipeline.py`, `cli.py`, and `autophotoeditor/api/app.py` through the facade.
- **Functions:** `local_name`, `parse_xmp`, `parse_curve_points_text`, `parse_point_curve`, `normalize_curve_points`.

### `apply_preset/color.py`
- **Description:** Converts between sRGB, linear RGB, OKLab, and OKLCH.
- **Use it when:** Performing perceptual color adjustments.
- **Used by:** `adjustments.py`, `cli.py`, and `pipeline.py`.
- **Functions:** `srgb_to_linear`, `linear_to_srgb`, `rgb_to_oklab`, `oklab_to_rgb`, `rgb_to_oklch`, `oklch_to_rgb`.

### `apply_preset/io.py`
- **Description:** Loads, saves, and reports on preset-processing images.
- **Use it when:** Reading source images or writing processed/debug output.
- **Used by:** `pipeline.py`, `cli.py`, and the API facade.
- **Functions:** `load_image`, `save_image`, `save_debug_stage`, `image_statistics`.

### `apply_preset/adjustments.py`
- **Description:** Applies individual preset categories to an image.
- **Use it when:** Applying exposure, tone, HSL, color, effects, noise, or sharpening.
- **Used by:** `pipeline.py`, `cli.py`, and `autophotoeditor/api/app.py` through the facade.
- **Functions:** `apply_exposure`, `apply_white_balance`, `apply_tone`, `apply_point_curve`, `apply_hsl`, `apply_color`, `apply_clarity`, `apply_texture`, `apply_dehaze`, `apply_effects`, `apply_noise_reduction`, `apply_sharpening`.

### `apply_preset/cli.py`
- **Description:** Parses preset command-line options and builds selected processing stages.
- **Use it when:** Translating CLI flags or overrides into `PipelineOptions`.
- **Used by:** `pipeline.py` and direct CLI execution.
- **Functions:** `build_parser`, `any_processing_argument_requested`, `build_pipeline_options`, `apply_cli_overrides`, `print_settings`.

### `apply_preset/pipeline.py`
- **Description:** Coordinates preset parsing, selected adjustments, and output.
- **Use it when:** Running the complete preset workflow.
- **Used by:** `apply_preset/engine.py`, `apply_preset/cli.py`, and `autophotoeditor/api/app.py`.
- **Functions:** `process_image`, `main`.

### `apply_preset/__init__.py`
- **Description:** Public preset package exports.
- **Use it when:** Importing preset processing and its action modules.
- **Used by:** API code and application callers.
- **Functions:** Re-exports facade functions; no processing implementation.

## Automatic Enhancement

### `auto_enhance/engine.py`
- **Description:** Compatibility facade for automatic enhancement and image analysis.
- **Use it when:** Preserving legacy imports.
- **Used by:** `auto_enhance/__init__.py` and `autophotoeditor/api/app.py`.
- **Functions:** Re-exports `enhance`, `extract_image_data`, recommendation, I/O, and analysis functions.

### `auto_enhance/core.py`
- **Description:** Shared enhancement configuration, image helpers, mask loading, and core enhancement operation.
- **Use it when:** Applying an analysis result to an image or using common analysis utilities.
- **Used by:** `analysis.py`, `recommendations.py`, `pipeline.py`, `enhancement.py`, and `io.py`.
- **Functions:** `enhance`, `_apply_auto_lighting`, `_apply_local_neutral_tone`, `_apply_library_luminosity`, `load_external_masks`, `resize_longest`, `ensure_mask`, `srgb_to_linear`, `linear_luminance`.

### `auto_enhance/analysis.py`
- **Description:** Computes scene, subject, tone, white-balance, color, contrast, detail, and region measurements.
- **Use it when:** Building an analysis report before enhancement.
- **Used by:** `pipeline.py`, `recommendations.py`, and `autophotoeditor/api/app.py` through the facade.
- **Functions:** `derive_subject_mask`, `estimate_semantic_color_masks`, `analyze_tone`, `analyze_white_balance`, `analyze_color_profile`, `analyze_contrast`, `analyze_detail`, `analyze_mask_region`, `analyze_grid_regions`, `classify_scene`.

### `auto_enhance/recommendations.py`
- **Description:** Converts measured image properties into edit recommendations.
- **Use it when:** Creating exposure and edit settings for the enhancer.
- **Used by:** `pipeline.py` and the API analysis route.
- **Functions:** `derive_exposure_recommendation`, `derive_edit_recommendations`.

### `auto_enhance/enhancement.py`
- **Description:** Public enhancement entry point and configuration types.
- **Use it when:** Applying automatic corrections to a BGR image.
- **Used by:** `auto_enhance/engine.py`, `auto_enhance/__init__.py`, and `autophotoeditor/api/app.py`.
- **Functions/classes:** `enhance`, `EnhancementConfig`, `EnhancementError`.

### `auto_enhance/io.py`
- **Description:** Public image-analysis output helpers.
- **Use it when:** Loading masks or saving analysis JSON/images.
- **Used by:** `pipeline.py`, the API, and compatibility imports.
- **Functions:** `load_external_masks`, `save_json`, `_save_image`.

### `auto_enhance/pipeline.py`
- **Description:** Coordinates image classification, analysis, recommendations, and JSON output.
- **Use it when:** Running the complete automatic enhancement analysis workflow.
- **Used by:** `cli.py`, `engine.py`, and `autophotoeditor/api/app.py`.
- **Functions:** `classify_image`, `extract_image_data`, `save_json`, `build_parser`, `main`.

### `auto_enhance/cli.py` and `auto_enhance/__init__.py`
- **Description:** CLI entry point and public package facade.
- **Use it when:** Running or importing automatic enhancement.
- **Used by:** Application/API callers and terminal launchers.
- **Functions:** `main`, `build_parser`; `__init__.py` re-exports the public engine API.

## Image Analysis V4

### `extract_image_data/engine.py`
- **Description:** Compatibility facade for the V4 image-analysis engine.
- **Use it when:** Preserving imports like `from ...extract_image_data import analyze_image`.
- **Used by:** `extract_image_data/__init__.py` and `autophotoeditor/api/app.py`.
- **Functions:** Re-exports the image-analysis pipeline and support functions.

### `extract_image_data/shared.py`
- **Description:** Shared numeric, mask, color-conversion, sampling, and LAB helpers used by the image analysis code.
- **Use it when:** Turning the source image into luminance, neutral-tone, and exposure measurements.
- **Used by:** All V4 analysis modules.
- **Functions:** `clamp`, `safe_float`, `percentile`, `normalized_mask`, `union_masks`, `bgr_to_rgb_float`, `srgb_to_linear`, `rgb_to_linear_luminance`, `bgr_to_lab`, `sample_array`, and related helpers.

### `extract_image_data/masks.py`
- **Description:** Reads optional external masks and derives subject/semantic coverage only as contextual input for the analysis.
- **Use it when:** The image needs subject/scene context while computing luminance and neutral-tone recommendations.
- **Used by:** `pipeline.py` and the analysis flow.
- **Functions:** `decode_external_mask`, `load_masks`, `spectral_residual_saliency`, `derive_subject_mask`, `derive_semantic_masks`.

### `extract_image_data/scene.py`
- **Description:** Classifies overall scene type from luminance and semantic coverage to guide enhancement values.
- **Use it when:** Deciding portrait, landscape, low-key, high-key, or general enhancement targets.
- **Used by:** `pipeline.py`.
- **Functions:** `classify_scene`.

### `extract_image_data/tone.py`
- **Description:** Calculates luminance statistics, exposure recommendations, and Lightroom-style tone controls for the best luminosity output.
- **Use it when:** Measuring exposure and tonal range for the image.
- **Used by:** `pipeline.py` and `report.py`.
- **Functions:** `calculate_tone_stats`, `exposure_transform`, `derive_exposure_recommendation`, `derive_lightroom_tone_controls`.

### `extract_image_data/white_balance.py`
- **Description:** Finds neutral patches and estimates RGB white-balance gains for the best neutral tone.
- **Use it when:** Measuring and correcting color casts to reach neutral balance.
- **Used by:** `pipeline.py` and `report.py`.
- **Functions:** `find_neutral_patches`, `gray_world_gains`, `shades_of_gray_gains`, `optimize_neutral_lab_gains`, `analyze_white_balance`, `apply_rgb_gains`.

### `extract_image_data/color.py`
- **Description:** Measures saturation, detail, histogram, and grid-region information for the image-analysis report.
- **Use it when:** Building color/detail diagnostics used for enhancement values.
- **Used by:** `pipeline.py`.
- **Functions:** `analyze_color`, `analyze_detail`, `calculate_histogram`, `analyze_grid`.

### `extract_image_data/report.py`
- **Description:** Builds image reports, lighting diagnostics, correction plans, and quality scores.
- **Use it when:** Producing the final JSON analysis and enhancement recommendation output.
- **Used by:** `pipeline.py`.
- **Functions:** `analyze_region`, `analyze_tone`, `analyze_lighting_quality`, `derive_edit_recommendations`, `derive_correction_plan`, `analyze_quality`.

### `extract_image_data/pipeline.py`, `cli.py`, and `__init__.py`
- **Description:** `pipeline.py` reads the image and produces enhancement values for luminosity and neutral tone; `cli.py` handles terminal arguments; `__init__.py` exposes the public API.
- **Use it when:** Running `analyze_image`, `analyze`, `process`, or the analyzer CLI to get image enhancement recommendations.
- **Used by:** `autophotoeditor/api/app.py`, compatibility imports, and terminal launchers.
- **Functions:** `analyze_image`, `analyze`, `process`, `load_image`, `build_parser`, `main`.

> Summary: `extract_image_data` is not a masking engine. It reads the image and computes the values needed for the best luminosity and best neutral tone.

## Semantic Mask Extraction

### `extract_mask_data/engine.py`
- **Description:** Compatibility facade for semantic mask analysis and enhancement planning.
- **Use it when:** Preserving imports from the former monolithic mask analyzer.
- **Used by:** `extract_mask_data/__init__.py` and `autophotoeditor/api/app.py`.
- **Functions:** Re-exports `run_pipeline`, mask analysis, and adjustment functions.

### `extract_mask_data/shared.py`
- **Description:** Logging, device selection, color science, and statistical helpers.
- **Use it when:** Supporting semantic mask calculations.
- **Used by:** All other mask-extraction modules.
- **Functions:** `configure_logging`, `choose_device`, `weighted_percentile`, `prepare_color_data`, `linear_rgb_to_lab`, `luminance_from_linear_rgb`.

### `extract_mask_data/masks.py`
- **Description:** Loads mask JSON/files and calculates region, global, and surrounding-mask statistics.
- **Use it when:** Reading masks and measuring their image context.
- **Used by:** `analysis.py`, `profiles.py`, `enhancement.py`, and `pipeline.py`.
- **Functions:** `resolve_mask_path`, `load_mask_file`, `load_masks_from_json`, `get_weighted_pixels`, `mask_values`, `create_surrounding_mask`, `surrounding_ring`, `calculate_region_stats`, `calculate_global_stats`.

### `extract_mask_data/analysis.py`
- **Description:** Calculates per-mask white balance, exposure, tone, contrast, color, clarity, and dehaze adjustments.
- **Use it when:** Deriving recommended edits for one semantic mask.
- **Used by:** `profiles.py`, `enhancement.py`, and `pipeline.py`.
- **Functions:** `estimate_white_balance`, `exposure_from_luminance`, `calculate_tone_adjustments`, `calculate_contrast`, `calculate_color_adjustments`, `calculate_clarity`, `calculate_dehaze`, `context_exposure_adjustment`, `context_white_balance`.

### `extract_mask_data/profiles.py`
- **Description:** Applies limits, strength, and mask-type-specific safety rules.
- **Use it when:** Converting raw recommendations into safe mask edits.
- **Used by:** `enhancement.py` and `pipeline.py`.
- **Functions:** `limit_values`, `apply_strength`, `adjust_for_mask_type`.

### `extract_mask_data/enhancement.py`
- **Description:** Builds global mask enhancement results and serializes extracted masks.
- **Use it when:** Creating output enhancement data from mask analysis.
- **Used by:** `pipeline.py` and the API.
- **Functions:** `calculate_global_enhancement`, `smooth_mask_values`, `build_output`, `_extract_model_masks`.

### `extract_mask_data/pipeline.py`, `cli.py`, and `__init__.py`
- **Description:** `pipeline.py` runs mask extraction and report generation; `cli.py` validates command-line arguments; `__init__.py` exposes the package.
- **Use it when:** Running semantic mask extraction end to end.
- **Used by:** `autophotoeditor/api/app.py` and terminal launchers.
- **Functions:** `run_pipeline`, `run`, `save_json`, `print_summary`, `build_parser`, `validate_args`, `main`.

## Geometry Correction

### `geometry_correction/engine.py` and `__init__.py`
- **Description:** Compatibility facade for geometry correction.
- **Use it when:** Importing geometry correction through the stable package API.
- **Used by:** `autophotoeditor/api/app.py` and geometry tests.
- **Functions:** Re-exports detection, transform, canvas, I/O, and CLI functions.

### `geometry_correction/core.py`
- **Description:** Geometry data classes and correction constants.
- **Use it when:** Defining line, vanishing-point, and geometry estimate data.
- **Used by:** All geometry action modules.
- **Functions/classes:** `DetectedLine`, `VanishingPointResult`, `GeometryEstimate`.

### `geometry_correction/io.py`
- **Description:** Reads and writes geometry-processing images.
- **Use it when:** Handling geometry input/output files.
- **Used by:** `cli.py`, the API, and the package facade.
- **Functions:** `read_image`, `write_image`.

### `geometry_correction/detection.py`
- **Description:** Detects lines, orientations, level, and vanishing points.
- **Use it when:** Measuring perspective or rotation problems.
- **Used by:** `transforms.py` and `cli.py`.
- **Functions:** `detect_lines`, `split_orientation`, `estimate_level`, `robust_vanishing_point`, `perspective_amount`.

### `geometry_correction/transforms.py`
- **Description:** Builds rotation, perspective, scale, and homography transforms.
- **Use it when:** Creating the correction matrix for an image.
- **Used by:** `canvas.py`, `cli.py`, the API, and geometry tests.
- **Functions:** `analyze_geometry`, `vp_projective_transform`, `rotation_homography`, `build_transform`, `validate_homography`.

### `geometry_correction/canvas.py`
- **Description:** Fits transformed images to a canvas, renders validity masks, crops, and draws debug overlays.
- **Use it when:** Rendering the corrected image after a transform is built.
- **Used by:** `cli.py` and `autophotoeditor/api/app.py`.
- **Functions:** `transformed_corners`, `fit_to_canvas`, `render_validity_mask`, `crop_using_mask`, `draw_debug`.

### `geometry_correction/cli.py`
- **Description:** Command-line geometry correction workflow.
- **Use it when:** Running geometry correction from a terminal.
- **Used by:** Direct CLI launchers.
- **Functions:** `parse_args`, `main`.

## Lens Correction

### `lens_correction/engine.py` and `__init__.py`
- **Description:** Compatibility facade for lens correction.
- **Use it when:** Importing lens correction through the stable package API.
- **Used by:** `autophotoeditor/api/app.py` and application callers.
- **Functions:** Re-exports profile, correction, and CLI functions.

### `lens_correction/core.py`
- **Description:** Shared lens data types and numeric/string helpers.
- **Use it when:** Handling lens correction options and normalized values.
- **Used by:** `profiles.py`, `correction.py`, and `cli.py`.
- **Functions/classes:** `ImageMetadata`, `ManualDistortion`, `LensCorrectionOptions`, `safe_float`, `normalize_string`, `finite_image`.

### `lens_correction/profiles.py`
- **Description:** Reads image metadata and searches Lensfun camera/lens profiles.
- **Use it when:** Selecting automatic optical correction data.
- **Used by:** `correction.py`, `cli.py`, and the API.
- **Functions:** `load_image`, `save_image`, `read_exif`, `require_lensfun`, `create_database`, `find_camera`, `find_lenses`, `find_lensfun_profile`, `print_lens_profile`.

### `lens_correction/correction.py`
- **Description:** Applies Lensfun or manual distortion correction and determines valid cropping.
- **Use it when:** Correcting barrel/pincushion distortion, TCA, or vignetting.
- **Used by:** `cli.py` and `autophotoeditor/api/app.py`.
- **Functions:** `apply_lensfun_distortion`, `estimate_focal_length`, `estimate_aperture`, `apply_manual_distortion`, `calculate_valid_crop`, `crop_to_valid_area`.

### `lens_correction/cli.py`
- **Description:** Coordinates profile selection, correction, debug-grid generation, and command-line execution.
- **Use it when:** Running a complete lens-correction operation.
- **Used by:** `lens_correction/engine.py`, the package facade, and direct CLI launchers.
- **Functions:** `process`, `create_debug_grid`, `list_cameras`, `list_lenses`, `load_profile_from_cli`, `build_parser`, `main`.

## Manual Adjustment

### `manual_adjust/engine.py`
- **Description:** Applies direct tonal and color adjustment values, globally or through a mask.
- **Use it when:** The user supplies explicit exposure, contrast, saturation, temperature, or tint values.
- **Used by:** `autophotoeditor/api/app.py` and `manual_adjust/__init__.py`.
- **Functions:** `normalize_adjustments`, `apply`, `apply_masked`, `_apply_tonal_adjustments`, `_apply_color_adjustments`.

### `manual_adjust/__init__.py`
- **Description:** Public facade for manual adjustments.
- **Use it when:** Importing manual adjustment operations.
- **Used by:** API callers and tests.
- **Functions:** Re-exports `apply` and `apply_masked`.

## Masked Adaptive Enhancement

### `masked_adaptive_enhance/engine.py` and `__init__.py`
- **Description:** Compatibility facade for localized enhancement using semantic masks.
- **Use it when:** Preserving old imports or accessing the full masked workflow.
- **Used by:** `autophotoeditor/api/app.py` and masked-enhancement tests.
- **Functions:** Re-exports `process`, `process_mask_set`, `enhance_region`, and related actions.

### `masked_adaptive_enhance/core.py`
- **Description:** Global configuration, mask profiles, and shared constants.
- **Use it when:** Defining enhancement strength, protection, feathering, and mask behavior.
- **Used by:** All masked-enhancement action modules.
- **Functions/classes:** `GlobalConfig`, `MaskProfile`, `MASK_PROFILES`, `CONFIG`.

### `masked_adaptive_enhance/io.py`
- **Description:** Handles Unicode image I/O, mask discovery, loading, cleaning, and feathering.
- **Use it when:** Reading source images or preparing masks.
- **Used by:** `analysis.py`, `region.py`, `pipeline.py`, and `cli.py`.
- **Functions:** `imread_unicode`, `imwrite_unicode`, `resolve_path`, `extract_mask_entries`, `discover_masks_from_directory`, `load_mask`, `clean_mask`, `feather_mask`.

### `masked_adaptive_enhance/analysis.py`
- **Description:** Measures masked regions and their surrounding context, including neutral gains.
- **Use it when:** Deriving local corrections from a mask and its neighboring pixels.
- **Used by:** `color.py`, `effects.py`, `region.py`, and `pipeline.py`.
- **Functions:** `weighted_percentile`, `weighted_mean`, `analyze_region`, `mask_bbox`, `surrounding_ring`, `analyze_surrounding`, `estimate_global_neutral_gains`, `estimate_local_neutral_gains`.

### `masked_adaptive_enhance/color.py`
- **Description:** Computes and applies local neutral/color and tonal-range corrections.
- **Use it when:** Correcting white balance, exposure, shadows, and highlights inside a mask.
- **Used by:** `effects.py`, `region.py`, and `pipeline.py`.
- **Functions:** `apply_neutral_color`, `apply_subtle_lab_neutralization`, `calculate_exposure`, `calculate_shadow_lift`, `calculate_highlight_reduction`.

### `masked_adaptive_enhance/effects.py`
- **Description:** Applies local tonal, clarity, vibrance, and sharpening effects.
- **Use it when:** Enhancing a corrected masked region.
- **Used by:** `region.py` and `pipeline.py`.
- **Functions:** `apply_tonal_adjustment`, `apply_local_clarity`, `apply_vibrance`, `apply_luminance_sharpening`.

### `masked_adaptive_enhance/region.py`
- **Description:** Combines context-aware corrections into one enhanced region and integrates mask edges.
- **Use it when:** Processing a single mask region.
- **Used by:** `pipeline.py` and the API.
- **Functions:** `integrate_mask_with_surrounding`, `enhance_region`.

### `masked_adaptive_enhance/pipeline.py`
- **Description:** Sorts masks, processes mask sets, composites results, and coordinates the complete workflow.
- **Use it when:** Running masked enhancement over one or more masks.
- **Used by:** `cli.py`, the facade, and `autophotoeditor/api/app.py`.
- **Functions:** `composite_mask`, `mask_sort_key`, `discover_person_masks`, `process_mask_set`, `process`.

### `masked_adaptive_enhance/cli.py`
- **Description:** Command-line entry point for masked adaptive enhancement.
- **Use it when:** Running localized enhancement from a terminal.
- **Used by:** Direct CLI launchers.
- **Functions:** `build_parser`, `main`.

## Choosing A Module

- Need to **run a complete operation**: use `pipeline.py` or the package facade.
- Need to **call one processing action**: import the matching action module.
- Need to **preserve an old import**: use the package root or `engine.py` facade.
- Need to **add a new operation**: place it beside the closest action module and expose it through the package facade only when it becomes public.
