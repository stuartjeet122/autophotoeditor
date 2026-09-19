from __future__ import annotations

from .core import *

def spectral_residual_saliency(bgr: np.ndarray) -> np.ndarray:
    """OpenCV-only spectral residual saliency fallback."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    small_w = max(32, min(128, w))
    small_h = max(32, min(128, h))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    dft = cv2.dft(small, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
    log_mag = np.log(mag + EPS)
    avg = cv2.boxFilter(log_mag, -1, (3, 3), normalize=True)
    residual = log_mag - avg
    exp_mag = np.exp(residual)
    real, imag = cv2.polarToCart(exp_mag, phase)
    spec = np.dstack([real, imag]).astype(np.float32)
    inv = cv2.idft(spec, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    sal = inv * inv
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)
    sal -= float(sal.min())
    maxv = float(sal.max())
    if maxv > EPS:
        sal /= maxv
    return np.clip(sal, 0.0, 1.0).astype(np.float32)


def center_prior(shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    sx = max(w * 0.32, 1.0)
    sy = max(h * 0.32, 1.0)
    g = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    return g.astype(np.float32)


def derive_subject_mask(
    bgr: np.ndarray,
    external_masks: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    h, w = bgr.shape[:2]

    for key in ("subject", "person", "foreground", "main_subject"):
        if key in external_masks:
            m = ensure_mask(external_masks[key], (h, w))
            cov = mask_coverage(m)
            conf = clamp(0.78 + min(cov, 40.0) / 200.0, 0.0, 0.98)
            return m, m.astype(np.float32), f"external:{key}", conf

    sal = spectral_residual_saliency(bgr)
    prior = center_prior((h, w))
    combined = 0.72 * sal + 0.28 * prior

    threshold = max(float(np.percentile(combined, 72)), 0.38)
    mask = combined >= threshold

    # Clean small isolated patches.
    u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = u8 > 0

    cov = mask_coverage(mask)
    sal_strength = float(np.mean(combined[mask])) if np.any(mask) else 0.0
    conf = clamp(0.35 + 0.45 * sal_strength + 0.2 * (1.0 if 3.0 <= cov <= 65.0 else 0.3), 0.1, 0.82)
    return mask, combined.astype(np.float32), "saliency", conf


# ============================================================
# SEMANTIC COLOR HEURISTICS
# ============================================================

def estimate_semantic_color_masks(bgr: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Lightweight semantic-ish masks used only as supporting evidence.
    They are deliberately conservative and are NOT treated as ground truth.
    """
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    L = lab[:, :, 0]
    A = lab[:, :, 1].astype(np.int16) - 128
    B = lab[:, :, 2].astype(np.int16) - 128

    # Vegetation: mostly green/cyan-green and reasonably saturated.
    vegetation = (H >= 30) & (H <= 95) & (S >= 45) & (V >= 30)

    # Sky: blue/cyan, typically top-biased; this is supporting evidence only.
    h, w = H.shape
    yy = np.arange(h)[:, None]
    top_bias = yy < int(h * 0.75)
    sky = (H >= 85) & (H <= 135) & (S >= 30) & (V >= 70) & top_bias

    # Skin heuristic in YCrCb is usually more robust than raw HSV alone.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    skin = (
        (Y >= 35)
        & (Cr >= 133) & (Cr <= 180)
        & (Cb >= 75) & (Cb <= 135)
        & (S >= 15) & (S <= 190)
    )

    # Neutral candidates - broad; stricter scoring happens in WB analysis.
    neutral = (S <= 50) & (V >= 40) & (V <= 245) & (L >= 35) & (L <= 245)

    # Warm-source-colored pixels, useful to avoid over-correcting intentional tungsten/neon.
    warm = (B >= 10) & (A >= -5) & (V >= 45)

    return {
        "vegetation": vegetation,
        "sky": sky,
        "skin": skin,
        "neutral_candidates": neutral,
        "warm_pixels": warm,
    }


# ============================================================
# TONE / EXPOSURE
# ============================================================

def analyze_tone(
    bgr: np.ndarray,
    lab: np.ndarray,
    linear_y: np.ndarray,
    subject_mask: np.ndarray,
    saliency_map: np.ndarray,
) -> Dict[str, Any]:
    L = lab[:, :, 0].astype(np.float32)

    percentiles = percentile_dict(
        L,
        (0.1, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.9),
    )
    p01 = percentiles["p1"] if "p1" in percentiles else percentiles.get("p1_0", 0.0)
    p99 = percentiles["p99"] if "p99" in percentiles else percentiles.get("p99_0", 255.0)

    any_high_clip = np.any(bgr >= 254, axis=2)
    all_high_clip = np.all(bgr >= 254, axis=2)
    any_low_clip = np.any(bgr <= 1, axis=2)
    all_low_clip = np.all(bgr <= 1, axis=2)

    # Linear-light exposure evidence.
    y = np.clip(linear_y.astype(np.float32), 0.0, 1.0)
    ev = luminance_to_ev(y)

    valid = (y > 0.001) & (y < 0.98)
    global_ev = robust_median(ev[valid], -3.0)

    subject_valid = subject_mask & valid
    subject_ev = robust_median(ev[subject_valid], global_ev)

    # Saliency-weighted luminance provides an intent-aware middle ground.
    sw = np.clip(saliency_map.astype(np.float64), 0.0, 1.0)
    valid_w = sw * valid.astype(np.float64)
    if float(valid_w.sum()) > EPS:
        weighted_y = float(np.sum(y * valid_w) / np.sum(valid_w))
        salient_ev = math.log2(max(weighted_y, 1e-6) / 0.18)
    else:
        salient_ev = global_ev

    # Desired subject midpoint is intentionally conservative. We are not assuming
    # every scene should average to 18% gray.
    subject_cov = mask_coverage(subject_mask) / 100.0
    global_mid = float(np.median(y))
    subject_mid = float(np.median(y[subject_mask])) if np.any(subject_mask) else global_mid

    # Highlight headroom in linear light.
    y99 = float(np.percentile(y, 99.0))
    y999 = float(np.percentile(y, 99.9))
    highlight_headroom_ev = math.log2(1.0 / max(y99, 1e-6))

    mean_l = float(np.mean(L))
    median_l = float(np.median(L))
    std_l = float(np.std(L))
    dark_pct = float(np.mean(L <= 25) * 100.0)
    very_dark_pct = float(np.mean(L <= 10) * 100.0)
    bright_pct = float(np.mean(L >= 230) * 100.0)
    very_bright_pct = float(np.mean(L >= 245) * 100.0)
    high_any_pct = float(np.mean(any_high_clip) * 100.0)
    high_all_pct = float(np.mean(all_high_clip) * 100.0)
    low_any_pct = float(np.mean(any_low_clip) * 100.0)
    low_all_pct = float(np.mean(all_low_clip) * 100.0)

    return {
        # v2 compatibility fields
        "mean_luminance": round(mean_l, 4),
        "median_luminance": round(median_l, 4),
        "contrast_rms": round(std_l, 4),
        "percentiles": percentiles,
        "effective_dynamic_range": round(float(p99 - p01), 4),
        "dark_pixels_percentage": round(dark_pct, 4),
        "very_dark_pixels_percentage": round(very_dark_pct, 4),
        "bright_pixels_percentage": round(bright_pct, 4),
        "very_bright_pixels_percentage": round(very_bright_pct, 4),
        "highlight_clipping": {
            "any_channel_percentage": round(high_any_pct, 5),
            "all_channels_percentage": round(high_all_pct, 5),
        },
        "shadow_clipping": {
            "any_channel_percentage": round(low_any_pct, 5),
            "all_channels_percentage": round(low_all_pct, 5),
        },
        "is_clipped_highlights": high_all_pct > 0.5,
        "is_clipped_shadows": low_all_pct > 0.5,

        "perceptual": {
            "mean_luminance_lab": round(float(np.mean(L)), 4),
            "median_luminance_lab": round(float(np.median(L)), 4),
            "contrast_rms_lab": round(float(np.std(L)), 4),
            "percentiles_lab": percentiles,
            "effective_dynamic_range_lab": round(float(p99 - p01), 4),
            "dark_pixels_percentage": round(float(np.mean(L <= 25) * 100.0), 4),
            "very_dark_pixels_percentage": round(float(np.mean(L <= 10) * 100.0), 4),
            "bright_pixels_percentage": round(float(np.mean(L >= 230) * 100.0), 4),
            "very_bright_pixels_percentage": round(float(np.mean(L >= 245) * 100.0), 4),
        },
        "linear_light": {
            "mean_luminance": round(float(np.mean(y)), 6),
            "median_luminance": round(float(np.median(y)), 6),
            "global_median_ev_from_18pct": round(global_ev, 4),
            "subject_median_ev_from_18pct": round(subject_ev, 4),
            "saliency_weighted_ev_from_18pct": round(salient_ev, 4),
            "global_median": round(global_mid, 6),
            "subject_median": round(subject_mid, 6),
            "subject_coverage_percentage": round(subject_cov * 100.0, 4),
            "p99_luminance": round(y99, 6),
            "p99_9_luminance": round(y999, 6),
            "highlight_headroom_ev_at_p99": round(highlight_headroom_ev, 4),
        },
        "clipping": {
            "highlight_any_channel_percentage": round(float(np.mean(any_high_clip) * 100.0), 5),
            "highlight_all_channels_percentage": round(float(np.mean(all_high_clip) * 100.0), 5),
            "shadow_any_channel_percentage": round(float(np.mean(any_low_clip) * 100.0), 5),
            "shadow_all_channels_percentage": round(float(np.mean(all_low_clip) * 100.0), 5),
        },
    }


# ============================================================
# WHITE BALANCE
# ============================================================

def _channel_gains_from_estimated_illuminant(rgb: np.ndarray) -> Tuple[float, float, float]:
    r, g, b = [max(float(x), 1e-6) for x in rgb]
    target = (r + g + b) / 3.0
    return target / r, target / g, target / b


def _wb_estimator_gray_world(rgb_linear: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, float]:
    pixels = rgb_linear[valid]
    if pixels.shape[0] < 100:
        return np.array([1.0, 1.0, 1.0], np.float32), 0.0
    illum = np.mean(pixels, axis=0)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)
    spread = float(np.std(illum) / max(np.mean(illum), EPS))
    conf = clamp(0.35 + 0.35 * min(pixels.shape[0] / 10000.0, 1.0) - 0.15 * spread, 0.15, 0.7)
    return gains, conf


def _wb_estimator_shades_of_gray(rgb_linear: np.ndarray, valid: np.ndarray, p: float = 6.0) -> Tuple[np.ndarray, float]:
    pixels = rgb_linear[valid]
    if pixels.shape[0] < 100:
        return np.array([1.0, 1.0, 1.0], np.float32), 0.0
    illum = np.power(np.mean(np.power(np.maximum(pixels, 1e-6), p), axis=0), 1.0 / p)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)
    conf = clamp(0.32 + 0.35 * min(pixels.shape[0] / 10000.0, 1.0), 0.15, 0.67)
    return gains, conf


def _wb_estimator_neutral_lab(
    rgb_linear: np.ndarray,
    lab: np.ndarray,
    neutral_mask: np.ndarray,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    A = lab[:, :, 1].astype(np.float32) - 128.0
    B = lab[:, :, 2].astype(np.float32) - 128.0
    count = int(np.count_nonzero(neutral_mask))
    total = neutral_mask.size
    coverage = count / max(total, 1)

    if count < max(100, int(total * 0.0015)):
        return (
            np.array([1.0, 1.0, 1.0], np.float32),
            0.0,
            {"neutral_a": 0.0, "neutral_b": 0.0, "coverage_percentage": coverage * 100.0},
        )

    pixels = rgb_linear[neutral_mask]
    illum = np.median(pixels, axis=0)
    gains = np.asarray(_channel_gains_from_estimated_illuminant(illum), dtype=np.float32)

    neutral_a = robust_median(A[neutral_mask])
    neutral_b = robust_median(B[neutral_mask])
    neutrality = math.exp(-(abs(neutral_a) + abs(neutral_b)) / 22.0)
    coverage_score = min(coverage / 0.12, 1.0)
    conf = clamp(0.25 + 0.5 * coverage_score + 0.2 * neutrality, 0.2, 0.93)

    return gains, conf, {
        "neutral_a": neutral_a,
        "neutral_b": neutral_b,
        "coverage_percentage": coverage * 100.0,
    }


def gains_to_cast(gains: np.ndarray) -> Tuple[float, float]:
    """
    Maps RGB correction gains to stable relative warm/cool and green/magenta
    indicators. These are edit-direction indicators, not camera Kelvin values.
    """
    r, g, b = [max(float(x), 1e-6) for x in gains]
    # If correction needs more blue than red, source is warm => positive cast.
    temperature_cast = math.log2(b / r) * 12.0
    # If correction needs more green relative to R/B average, source is magenta.
    tint_cast = math.log2(g / math.sqrt(r * b)) * 12.0
    return temperature_cast, tint_cast


def analyze_white_balance(
    bgr: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    rgb_linear: np.ndarray,
    semantic_masks: Mapping[str, np.ndarray],
    external_masks: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    H, S, V = cv2.split(hsv)
    L = lab[:, :, 0]

    valid = (V >= 35) & (V <= 245) & (L >= 30) & (L <= 245)

    # Exclude colors that are poor neutral references.
    exclusion = np.zeros(valid.shape, dtype=bool)
    for key in ("skin", "vegetation", "sky"):
        if key in semantic_masks:
            exclusion |= semantic_masks[key]
        if key in external_masks:
            exclusion |= ensure_mask(external_masks[key], valid.shape)

    neutral_mask = valid & (S <= 48) & (~exclusion)

    gw_gains, gw_conf = _wb_estimator_gray_world(rgb_linear, valid & (~exclusion))
    sg_gains, sg_conf = _wb_estimator_shades_of_gray(rgb_linear, valid & (~exclusion), p=6.0)
    nl_gains, nl_conf, nl_meta = _wb_estimator_neutral_lab(rgb_linear, lab, neutral_mask)

    neutral_coverage = float(nl_meta.get("coverage_percentage", 0.0))
    # Neutral-pixel evidence is the most photographically meaningful source.
    # If there are almost no plausible neutrals, reduce confidence rather than
    # inventing a white point from a strongly colored scene.
    neutral_quality = clamp(neutral_coverage / 1.0, 0.0, 1.0)
    estimators = [
        ("neutral_lab", nl_gains, nl_conf * (0.55 + 0.45 * neutral_quality)),
        ("gray_world", gw_gains, gw_conf * 0.70),
        ("shades_of_gray", sg_gains, sg_conf * 0.70),
    ]

    valid_est = [(n, g, c) for n, g, c in estimators if c > 0.0]
    if valid_est:
        weights = np.array([c for _, _, c in valid_est], dtype=np.float64)
        log_gains = np.vstack([np.log(np.maximum(g, 1e-6)) for _, g, _ in valid_est])
        fused = np.exp(np.average(log_gains, axis=0, weights=weights)).astype(np.float32)

        # Agreement: how much estimators disagree in log-gain space.
        dispersion = float(np.mean(np.std(log_gains, axis=0))) if len(valid_est) > 1 else 0.0
        agreement = math.exp(-dispersion * 7.0)
        base_conf = float(np.average(weights, weights=weights))
        # Do not allow disagreement between estimators to turn into a strong
        # automatic white-balance correction. Intentional colored illumination is
        # common in real photographs.
        fused_conf = clamp(base_conf * (0.42 + 0.58 * agreement), 0.0, 0.96)
    else:
        fused = np.ones(3, dtype=np.float32)
        fused_conf = 0.0
        agreement = 0.0

    # Normalize gains around green for readability.
    fused = fused / max(float(fused[1]), 1e-6)
    temp_cast, tint_cast = gains_to_cast(fused)

    return {
        "method": "multi-estimator-fusion",
        "correction_gains_rgb": [round(float(x), 6) for x in fused],
        "estimated_temperature_cast": Estimate(temp_cast, fused_conf).to_json(),
        "estimated_tint_cast": Estimate(tint_cast, fused_conf).to_json(),
        "neutral_pixel_coverage_percentage": round(float(nl_meta["coverage_percentage"]), 4),
        "neutral_lab_a_median": round(float(nl_meta["neutral_a"]), 4),
        "neutral_lab_b_median": round(float(nl_meta["neutral_b"]), 4),
        "estimator_agreement": round(float(agreement), 4),
        "estimators": {
            name: {
                "gains_rgb": [round(float(x), 6) for x in gains / max(float(gains[1]), 1e-6)],
                "confidence": round(float(conf), 4),
            }
            for name, gains, conf in estimators
        },
        "interpretation": {
            "temperature": "warm" if temp_cast > 2.0 else "cool" if temp_cast < -2.0 else "neutral",
            "tint": "magenta" if tint_cast > 2.0 else "green" if tint_cast < -2.0 else "neutral",
        },
        "warning": "Temperature/tint are image-cast indicators, not camera Kelvin/tint metadata.",
    }


def analyze_color_profile(
    bgr: np.ndarray,
    hsv: np.ndarray,
    wb: Mapping[str, Any],
) -> Dict[str, Any]:
    B = bgr[:, :, 0].astype(np.float32)
    G = bgr[:, :, 1].astype(np.float32)
    R = bgr[:, :, 2].astype(np.float32)
    S = hsv[:, :, 1]

    sat_percentiles = percentile_dict(S, (5, 10, 25, 50, 75, 90, 95, 99))
    mean_b, mean_g, mean_r = float(np.mean(B)), float(np.mean(G)), float(np.mean(R))

    # Compatibility representation: scalar cast values are kept here while the
    # richer confidence-aware estimates remain in top-level white_balance.
    compat_wb = {
        "neutral_pixel_coverage_percentage": wb["neutral_pixel_coverage_percentage"],
        "neutral_lab_a_median": wb["neutral_lab_a_median"],
        "neutral_lab_b_median": wb["neutral_lab_b_median"],
        "estimated_tint_cast": wb["estimated_tint_cast"]["value"],
        "estimated_temperature_cast": wb["estimated_temperature_cast"]["value"],
        "confidence": wb["estimated_temperature_cast"]["confidence"],
        "interpretation": wb["interpretation"],
    }

    return {
        "mean_bgr": [round(mean_b, 4), round(mean_g, 4), round(mean_r, 4)],
        "std_bgr": [
            round(float(np.std(B)), 4),
            round(float(np.std(G)), 4),
            round(float(np.std(R)), 4),
        ],
        "saturation_mean": round(float(np.mean(S)), 4),
        "saturation_std": round(float(np.std(S)), 4),
        "saturation_percentiles": sat_percentiles,
        "low_saturation_percentage": round(float(np.mean(S < 25) * 100.0), 4),
        "medium_saturation_percentage": round(float(np.mean((S >= 25) & (S < 100)) * 100.0), 4),
        "high_saturation_percentage": round(float(np.mean(S >= 180) * 100.0), 4),
        "extreme_saturation_percentage": round(float(np.mean(S >= 220) * 100.0), 4),
        "white_balance": compat_wb,
        "diagnostic_rgb_ratios": {
            "red_blue_ratio": round(mean_r / max(mean_b, EPS), 6),
            "red_green_ratio": round(mean_r / max(mean_g, EPS), 6),
            "blue_green_ratio": round(mean_b / max(mean_g, EPS), 6),
        },
    }


# ============================================================
# CONTRAST / DETAIL / NOISE
# ============================================================

def analyze_contrast(lab_l: np.ndarray, linear_y: np.ndarray) -> Dict[str, float]:
    L = lab_l.astype(np.float32)
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=15, sigmaY=15)
    local_difference = L - blurred

    y = np.clip(linear_y.astype(np.float32), 0.0, 1.0)
    log_y = np.log2(np.maximum(y, 1e-5))

    return {
        "global_std_lab": round(float(np.std(L)), 4),
        "local_std_lab": round(float(np.std(local_difference)), 4),
        "rms_lab": round(float(np.sqrt(np.mean(np.square(L - np.mean(L))))), 4),
        "log_luminance_std_ev": round(float(np.std(log_y)), 4),
    }


def analyze_detail(gray: np.ndarray, subject_mask: np.ndarray) -> Dict[str, Any]:
    analysis_gray = resize_longest(gray, DEFAULT_DETAIL_MAX_DIMENSION)
    h0, w0 = gray.shape
    h, w = analysis_gray.shape

    if (h, w) != (h0, w0):
        subject = cv2.resize(subject_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        subject = subject_mask

    gray_f = analysis_gray.astype(np.float32)

    lap = cv2.Laplacian(analysis_gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    tenengrad = float(np.mean(grad * grad))
    grad_median = float(np.median(grad))

    # Flat-region noise estimator using robust MAD of a high-pass residual.
    smooth_large = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.2, sigmaY=1.2)
    highpass = gray_f - smooth_large

    local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    local_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), sigmaX=2.5, sigmaY=2.5)
    local_var = np.maximum(local_sq - local_mean * local_mean, 0.0)

    flat_threshold = float(np.percentile(local_var, 35.0))
    flat = local_var <= flat_threshold
    flat &= (analysis_gray >= 20) & (analysis_gray <= 235)

    hp_flat = highpass[flat]
    sigma = 1.4826 * robust_mad(hp_flat) if hp_flat.size >= 100 else float(np.std(highpass))

    # Motion/defocus clues from directional gradient anisotropy and edge width proxy.
    abs_gx = float(np.mean(np.abs(gx)))
    abs_gy = float(np.mean(np.abs(gy)))
    anisotropy = abs(abs_gx - abs_gy) / max(abs_gx + abs_gy, EPS)

    edges = grad > np.percentile(grad, 82)
    edge_density = float(np.mean(edges))

    if np.any(subject):
        subject_sharpness = float(np.mean(grad[subject]))
    else:
        subject_sharpness = float(np.mean(grad))

    # Confidence favors enough flat pixels for noise and enough edges for sharpness.
    flat_cov = float(np.mean(flat))
    noise_conf = clamp(0.25 + min(flat_cov / 0.25, 1.0) * 0.7, 0.2, 0.95)
    sharp_conf = clamp(0.3 + min(edge_density / 0.18, 1.0) * 0.6, 0.25, 0.93)

    return {
        "analysis_width": int(w),
        "analysis_height": int(h),
        "laplacian_variance": round(lap_var, 4),
        "tenengrad": round(tenengrad, 4),
        "gradient_median": round(grad_median, 4),
        "subject_gradient_mean": round(subject_sharpness, 4),
        "edge_density": round(edge_density, 6),
        "gradient_anisotropy": round(float(anisotropy), 6),
        "noise_sigma_flat_regions": Estimate(float(sigma), noise_conf).to_json(),
        "flat_region_coverage_percentage": round(flat_cov * 100.0, 4),
        "sharpness_confidence": round(sharp_conf, 4),
    }


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return safe_float(-np.sum(p * np.log2(p)))


# ============================================================
# REGION / SEMANTIC ANALYSIS
# ============================================================

def analyze_mask_region(
    name: str,
    mask: np.ndarray,
    lab: np.ndarray,
    hsv: np.ndarray,
    linear_y: np.ndarray,
) -> Dict[str, Any]:
    m = mask.astype(bool)
    cov = mask_coverage(m)
    if not np.any(m):
        return {"name": name, "coverage_percentage": 0.0, "available": False}

    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    y = linear_y
    vals = y[m]

    return {
        "name": name,
        "available": True,
        "coverage_percentage": round(cov, 4),
        "mean_luminance_lab": round(float(np.mean(L[m])), 4),
        "median_luminance_lab": round(float(np.median(L[m])), 4),
        "mean_saturation": round(float(np.mean(S[m])), 4),
        "median_linear_luminance": round(float(np.median(vals)), 6),
        "median_ev_from_18pct": round(float(np.median(luminance_to_ev(vals))), 4),
        "p05_linear_luminance": round(float(np.percentile(vals, 5)), 6),
        "p95_linear_luminance": round(float(np.percentile(vals, 95)), 6),
    }


def analyze_grid_regions(image: np.ndarray, rows: int = 3, cols: int = 3) -> Dict[str, Any]:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    regions: Dict[str, Any] = {}

    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * h // rows, (row + 1) * h // rows
            x1, x2 = col * w // cols, (col + 1) * w // cols
            rl = L[y1:y2, x1:x2]
            rs = S[y1:y2, x1:x2]
            if rl.size == 0:
                continue
            key = f"r{row + 1}_c{col + 1}"
            regions[key] = {
                "row": row + 1,
                "column": col + 1,
                "mean_luminance": round(float(np.mean(rl)), 4),
                "median_luminance": round(float(np.median(rl)), 4),
                "contrast": round(float(np.std(rl)), 4),
                "mean_saturation": round(float(np.mean(rs)), 4),
                "dark_percentage": round(float(np.mean(rl < 25) * 100), 4),
                "bright_percentage": round(float(np.mean(rl > 230) * 100), 4),
            }
    return regions


# ============================================================
# SCENE UNDERSTANDING
# ============================================================

def classify_scene(
    semantic_masks: Mapping[str, np.ndarray],
    external_masks: Mapping[str, np.ndarray],
    subject_mask: np.ndarray,
    tone: Mapping[str, Any],
) -> Dict[str, Any]:
    def cov(name: str) -> float:
        if name in external_masks:
            return mask_coverage(external_masks[name])
        if name in semantic_masks:
            return mask_coverage(semantic_masks[name])
        return 0.0

    skin_cov = cov("skin")
    sky_cov = cov("sky")
    veg_cov = cov("vegetation")
    person_cov = cov("person")
    face_cov = cov("face")
    subject_cov = mask_coverage(subject_mask)

    med = float(tone["linear_light"]["global_median"])
    p99 = float(tone["linear_light"]["p99_luminance"])

    scores = {
        "portrait": clamp((max(person_cov, face_cov * 2.0, skin_cov) / 28.0), 0.0, 1.0),
        "landscape": clamp((sky_cov + veg_cov) / 45.0, 0.0, 1.0),
        "low_key": clamp((0.12 - med) / 0.12, 0.0, 1.0),
        "high_key": clamp((med - 0.32) / 0.35, 0.0, 1.0),
    }

    if max(scores.values()) < 0.28:
        scene_type = "general"
        confidence = 0.45
    else:
        scene_type = max(scores, key=scores.get)
        confidence = clamp(0.45 + 0.45 * scores[scene_type], 0.45, 0.9)

    return {
        "type": scene_type,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "evidence": {
            "subject_coverage_percentage": round(subject_cov, 4),
            "person_coverage_percentage": round(person_cov, 4),
            "face_coverage_percentage": round(face_cov, 4),
            "skin_coverage_percentage": round(skin_cov, 4),
            "sky_coverage_percentage": round(sky_cov, 4),
            "vegetation_coverage_percentage": round(veg_cov, 4),
            "global_median_linear_luminance": round(med, 6),
            "p99_linear_luminance": round(p99, 6),
        },
    }


# ============================================================
# OPTIONAL AI SCENE CLASSIFIER
# ============================================================

class OptionalAIAnalyzer:
    """
    Optional zero-shot CLIP scene classifier.

    It only activates when --ai is supplied and the required packages/model are
    available. The rest of the analyzer remains fully functional without it.

    Default model: openai/clip-vit-base-patch32
    """

    LABELS = [
        "portrait photo",
        "group photo",
        "landscape photo",
        "city photo",
        "indoor photo",
        "night photo",
        "food photo",
        "product photo",
        "pet photo",
        "sports photo",
        "architecture photo",
        "document photo",
        "macro photo",
    ]

    def __init__(self, enabled: bool, model_name: str):
        self.enabled = enabled
        self.model_name = model_name
        self._pipe = None
        self.error: Optional[str] = None

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self._pipe is not None:
            return True
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline(
                task="zero-shot-image-classification",
                model=self.model_name,
            )
            return True
        except Exception as exc:
            self.error = f"AI scene classifier unavailable: {exc}"
            logger.warning(self.error)
            return False

    def analyze(self, bgr: np.ndarray) -> Dict[str, Any]:
        if not self._load():
            return {
                "enabled": bool(self.enabled),
                "available": False,
                "model": self.model_name,
                "error": self.error,
            }
        try:
            from PIL import Image  # type: ignore
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            outputs = self._pipe(image, candidate_labels=self.LABELS)
            top = outputs[:5]
            return {
                "enabled": True,
                "available": True,
                "model": self.model_name,
                "top_labels": [
                    {"label": str(x["label"]), "score": round(float(x["score"]), 6)}
                    for x in top
                ],
            }
        except Exception as exc:
            return {
                "enabled": True,
                "available": False,
                "model": self.model_name,
                "error": str(exc),
            }


