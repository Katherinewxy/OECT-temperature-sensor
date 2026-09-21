"""Local extraction logic preserved from the original analysis scripts."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
EPS = 1e-12

def boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]

def collect_real_environment_records(real_dir: Path) -> pd.DataFrame:
    rows = []
    for transfer_path in sorted(real_dir.rglob("Device_*_transfer.csv")):
        device_id, repeat_id = parse_device_repeat(transfer_path)
        transient_path = transfer_path.with_name(transfer_path.name.replace("_transfer.csv", "_transient.csv"))
        if not transient_path.exists():
            continue
        transfer_df = pd.read_csv(transfer_path, encoding="utf-8-sig")
        transient_df = pd.read_csv(transient_path, encoding="utf-8-sig")
        vg = pd.to_numeric(transfer_df.iloc[:, 0], errors="coerce").to_numpy(float)
        time = pd.to_numeric(transient_df.iloc[:, 0], errors="coerce").to_numpy(float)

        transient_temp_cols = list(transient_df.columns[1:])
        if not transient_temp_cols:
            continue
        ref_col = max(
            transient_temp_cols,
            key=lambda c: float(np.nanpercentile(pd.to_numeric(transient_df[c], errors="coerce"), 99)
                                - np.nanpercentile(pd.to_numeric(transient_df[c], errors="coerce"), 1)),
        )
        ref_y = pd.to_numeric(transient_df[ref_col], errors="coerce").to_numpy(float)
        windows = detect_pulse_windows(time, ref_y)

        common_cols = [c for c in transfer_df.columns[1:] if c in transient_df.columns]
        for temp_col in common_cols:
            temperature, temperature_label = parse_temperature_label(temp_col)
            transfer_id = pd.to_numeric(transfer_df[temp_col], errors="coerce").to_numpy(float)
            transient_id = pd.to_numeric(transient_df[temp_col], errors="coerce").to_numpy(float)
            repeat_type = "real_environment" if repeat_id == 1 else "temperature_stage"
            row = {
                "dataset": "real_environment_data",
                "source": "real_environment_data",
                "folder": transfer_path.parent.name,
                "device_id": device_id,
                "device_group": f"real_D{device_id}",
                "repeat_id": repeat_id,
                "repeat_type": repeat_type,
                "temperature": temperature,
                "temperature_label": temperature_label,
                "sample_id": f"real_D{device_id}_rep{repeat_id}_{temperature_label}C",
                "transfer_path": str(transfer_path),
                "transient_path": str(transient_path),
            }
            row.update(extract_transfer_features_from_arrays(vg, transfer_id))
            row.update(extract_transient_features_from_arrays(time, transient_id, windows))
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No real environment transfer/transient pairs found under {real_dir}")
    return pd.DataFrame(rows)

def crossing_time(t: np.ndarray, y: np.ndarray, threshold: float, start: int, end: int, direction: str) -> float:
    if end <= start + 1:
        return float("nan")
    idx = np.arange(start, end)
    vals = y[idx]
    hits = np.where(vals >= threshold)[0] if direction == "up" else np.where(vals <= threshold)[0]
    if len(hits) == 0:
        return float("nan")
    j = int(hits[0])
    if j == 0:
        return float(t[idx[j]])
    i0 = int(idx[j - 1])
    i1 = int(idx[j])
    if abs(y[i1] - y[i0]) < EPS:
        return float(t[i1])
    frac = (threshold - y[i0]) / (y[i1] - y[i0])
    return float(t[i0] + frac * (t[i1] - t[i0]))

def detect_pulse_windows(t: np.ndarray, y_ref: np.ndarray) -> list[tuple[int, int]]:
    t, y_ref = finite_xy(t, y_ref)
    if len(t) < 100:
        return []
    order = np.argsort(t)
    t = t[order]
    y_ref = y_ref[order]
    y_s = smooth(y_ref, 31)
    low = float(np.nanpercentile(y_s, 5))
    high = float(np.nanpercentile(y_s, 99))
    amp = high - low
    if not np.isfinite(amp) or amp <= EPS:
        return []
    threshold = low + 0.15 * amp
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 3 else 0.001
    mask = y_s > threshold
    mask = fill_short_boolean_gaps(mask, max(1, int(round(0.03 / max(dt, EPS)))))
    windows = []
    for start, end in boolean_runs(mask):
        duration = float(t[min(end, len(t) - 1) - 1] - t[start])
        if duration >= 0.35:
            windows.append((start, min(end, len(t) - 1)))
    return windows

def extract_transfer_features_from_arrays(vg: np.ndarray, current: np.ndarray) -> dict[str, float]:
    vg, current = finite_xy(vg, current)
    abs_id = np.abs(current)
    q25, q75 = np.nanpercentile(abs_id, [25, 75])
    k = max(1, int(np.ceil(len(abs_id) * 0.10)))
    feats = {
        "transfer_abs_id_top10_mean": float(np.nanmean(np.sort(abs_id)[-k:])),
        "transfer_abs_id_max": float(np.nanmax(abs_id)),
        "transfer_abs_id_median": float(np.nanmedian(abs_id)),
        "transfer_abs_id_mean": float(np.nanmean(abs_id)),
        "transfer_abs_id_std": float(np.nanstd(abs_id, ddof=1)),
        "transfer_abs_id_iqr": float(q75 - q25),
        "transfer_abs_id_at_vg0": interp_abs_id_at_vg(vg, current, 0.0),
        "transfer_abs_id_at_vgmax": interp_abs_id_at_vg(vg, current, float(np.nanmax(vg))),
        "transfer_area_abs": transfer_area(vg, current),
        "transfer_hysteresis_current_ratio": transfer_hysteresis_current_ratio(vg, current),
    }
    feats.update(transfer_gm_stats(vg, current))
    return feats

def extract_transient_features_from_arrays(t: np.ndarray, y: np.ndarray, windows: list[tuple[int, int]]) -> dict[str, float]:
    rises, decays = transient_events_from_windows(t, y, windows)
    t, y = finite_xy(t, y)
    feats: dict[str, float] = {
        "transient_id_mean": float(np.nanmean(y)),
        "transient_id_std": float(np.nanstd(y, ddof=1)),
        "transient_id_iqr": float(np.nanpercentile(y, 75) - np.nanpercentile(y, 25)),
        "transient_abs_area": float(np.trapz(np.abs(y), t)) if len(t) > 2 else np.nan,
        "transient_rise_event_count": float(len(rises)),
        "transient_decay_event_count": float(len(decays)),
    }
    for field in ["rise_time", "amplitude", "baseline", "high_state", "pre_noise", "post_noise"]:
        feats.update(summarize_events("transient_rise", rises, field))
    for field in ["decay_time", "decay_area", "amplitude", "baseline", "high_state", "pre_noise", "post_noise"]:
        feats.update(summarize_events("transient_decay", decays, field))
    rise_amp = feats.get("transient_rise_amplitude_median", np.nan)
    feats["transient_rise_noise_ratio"] = safe_ratio(feats.get("transient_rise_pre_noise_median", np.nan), rise_amp)
    decay_amp = feats.get("transient_decay_amplitude_median", np.nan)
    feats["transient_decay_noise_ratio"] = safe_ratio(feats.get("transient_decay_post_noise_median", np.nan), decay_amp)
    return feats

def fill_short_boolean_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    false_runs = boolean_runs(~out)
    for start, end in false_runs:
        if start == 0 or end == len(out):
            continue
        if end - start <= max_gap:
            out[start:end] = True
    return out

def finite_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]

def interp_abs_id_at_vg(vg: np.ndarray, current: np.ndarray, target: float) -> float:
    vals = []
    for bx, by in split_transfer_branches(vg, np.abs(current)):
        xs, ys = unique_sorted_xy(bx, by)
        if len(xs) >= 3 and float(xs.min()) <= target <= float(xs.max()):
            vals.append(float(np.interp(target, xs, ys)))
    return float(np.nanmedian(vals)) if vals else float("nan")

def parse_device_repeat(path: Path) -> tuple[int, int]:
    match = re.search(r"Device_(\d+)-(\d+)_(?:transfer|transient)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse device/repeat from {path}")
    return int(match.group(1)), int(match.group(2))

def parse_temperature_label(label: str) -> tuple[float, str]:
    cleaned = str(label).replace("\ufeff", "").replace("℃", "").replace("°C", "").replace("C", "").strip()
    return float(cleaned), cleaned

def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < EPS:
        return float("nan")
    return float(num / den)

def smooth(values: np.ndarray, width: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < width:
        return arr
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(arr, kernel, mode="same")

def split_transfer_branches(vg: np.ndarray, current: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    vg, current = finite_xy(vg, current)
    if len(vg) < 5:
        return [(vg, current)]
    turn = int(np.nanargmax(vg))
    if 0 < turn < len(vg) - 1:
        return [(vg[: turn + 1], current[: turn + 1]), (vg[turn:], current[turn:])]
    return [(vg, current)]

def summarize_events(prefix: str, events: list[dict[str, float]], field: str) -> dict[str, float]:
    vals = np.asarray([e[field] for e in events if np.isfinite(e.get(field, np.nan))], dtype=float)
    if len(vals) == 0:
        return {f"{prefix}_{field}_median": np.nan, f"{prefix}_{field}_mean": np.nan, f"{prefix}_{field}_std": np.nan}
    return {
        f"{prefix}_{field}_median": float(np.nanmedian(vals)),
        f"{prefix}_{field}_mean": float(np.nanmean(vals)),
        f"{prefix}_{field}_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }

def transfer_area(vg: np.ndarray, current: np.ndarray) -> float:
    vals = []
    for bx, by in split_transfer_branches(vg, np.abs(current)):
        xs, ys = unique_sorted_xy(bx, by)
        if len(xs) >= 5:
            vals.append(float(np.trapz(ys, xs)))
    return float(np.nansum(vals)) if vals else float("nan")

def transfer_gm_stats(vg: np.ndarray, current: np.ndarray) -> dict[str, float]:
    gm_values = []
    gm_areas = []
    for bx, by in split_transfer_branches(vg, np.abs(current)):
        xs, ys = unique_sorted_xy(bx, by)
        if len(xs) < 5:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            gm = np.gradient(ys, xs)
        keep = np.isfinite(gm)
        if keep.sum() < 5:
            continue
        gm_values.append(gm[keep])
        gm_areas.append(float(np.trapz(np.abs(gm[keep]), xs[keep])))
    if not gm_values:
        return {"transfer_gm_area": np.nan, "transfer_gm_std": np.nan, "transfer_gm_median_abs": np.nan, "transfer_gm_max_abs": np.nan}
    merged = np.concatenate(gm_values)
    return {
        "transfer_gm_area": float(np.nansum(gm_areas)),
        "transfer_gm_std": float(np.nanstd(merged, ddof=1)),
        "transfer_gm_median_abs": float(np.nanmedian(np.abs(merged))),
        "transfer_gm_max_abs": float(np.nanmax(np.abs(merged))),
    }

def transfer_hysteresis_current_ratio(vg: np.ndarray, current: np.ndarray) -> float:
    branches = split_transfer_branches(vg, np.abs(current))
    if len(branches) < 2:
        return float("nan")
    xs0, ys0 = unique_sorted_xy(branches[0][0], branches[0][1])
    xs1, ys1 = unique_sorted_xy(branches[1][0], branches[1][1])
    if len(xs0) < 5 or len(xs1) < 5:
        return float("nan")
    low = max(float(xs0.min()), float(xs1.min()))
    high = min(float(xs0.max()), float(xs1.max()))
    if high <= low:
        return float("nan")
    grid = np.linspace(low, high, 160)
    fwd = np.interp(grid, xs0, ys0)
    rev = np.interp(grid, xs1, ys1)
    return safe_ratio(float(np.nanmedian(np.abs(fwd - rev))), float(np.nanmedian(np.abs(fwd))))

def transient_events_from_windows(
    t: np.ndarray,
    y: np.ndarray,
    windows: list[tuple[int, int]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    keep = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t[keep], dtype=float)
    y = np.asarray(y[keep], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if len(t) < 100 or not windows:
        return [], []
    y_s = smooth(y, 21)
    rises: list[dict[str, float]] = []
    decays: list[dict[str, float]] = []
    for idx, (start_idx, end_idx) in enumerate(windows):
        start_idx = int(np.clip(start_idx, 0, len(t) - 2))
        end_idx = int(np.clip(end_idx, start_idx + 2, len(t) - 1))
        edge_t = float(t[start_idx])
        decay_t = float(t[end_idx])
        next_start = int(windows[idx + 1][0]) if idx + 1 < len(windows) else len(t) - 1

        pre = (t >= edge_t - 0.25) & (t < edge_t - 0.03)
        if int(pre.sum()) < 8:
            pre = np.arange(len(t)) < start_idx
        post = (t >= max(edge_t + 0.20, decay_t - 0.30)) & (t <= max(edge_t + 0.24, decay_t - 0.05))
        if int(post.sum()) < 8:
            post = (np.arange(len(t)) >= start_idx) & (np.arange(len(t)) < end_idx)
        if int(pre.sum()) >= 8 and int(post.sum()) >= 8:
            baseline = float(np.nanmedian(y_s[pre]))
            high_state = float(np.nanmedian(y_s[post]))
            delta = high_state - baseline
            pre_noise = float(np.nanstd(y_s[pre], ddof=1)) if int(pre.sum()) > 2 else 0.0
            post_noise = float(np.nanstd(y_s[post], ddof=1)) if int(post.sum()) > 2 else 0.0
            if np.isfinite(delta) and delta > max(EPS, 4.0 * pre_noise):
                rise_start = int(np.searchsorted(t, edge_t - 0.35))
                y10 = baseline + 0.10 * delta
                y90 = baseline + 0.90 * delta
                t10 = crossing_time(t, y_s, y10, rise_start, end_idx, "up")
                t90 = crossing_time(t, y_s, y90, rise_start, end_idx, "up")
                rise_time = t90 - t10
                if np.isfinite(rise_time) and rise_time > 0 and rise_time < max(decay_t - edge_t, EPS):
                    rises.append(
                        {
                            "edge_time": edge_t,
                            "t10": t10,
                            "t90": t90,
                            "rise_time": float(rise_time),
                            "amplitude": float(delta),
                            "baseline": baseline,
                            "high_state": high_state,
                            "pre_noise": pre_noise,
                            "post_noise": post_noise,
                        }
                    )

        if end_idx >= len(t) - 3:
            continue
        high_pre = (t >= decay_t - 0.25) & (t < decay_t - 0.03)
        post_end_idx = max(end_idx + 2, min(next_start, len(t) - 1))
        post_decay = (t >= decay_t + 0.04) & (t <= min(float(t[post_end_idx]), decay_t + 0.35))
        if int(post_decay.sum()) < 8:
            post_decay = (np.arange(len(t)) > end_idx) & (np.arange(len(t)) < post_end_idx)
        if int(high_pre.sum()) < 8 or int(post_decay.sum()) < 8:
            continue
        high_state = float(np.nanmedian(y_s[high_pre]))
        baseline = float(np.nanmedian(y_s[post_decay]))
        amp = high_state - baseline
        pre_noise = float(np.nanstd(y_s[high_pre], ddof=1)) if int(high_pre.sum()) > 2 else 0.0
        post_noise = float(np.nanstd(y_s[post_decay], ddof=1)) if int(post_decay.sum()) > 2 else 0.0
        if not np.isfinite(amp) or amp <= max(EPS, 4.0 * post_noise):
            continue
        y90 = baseline + 0.90 * amp
        y10 = baseline + 0.10 * amp
        t90 = crossing_time(t, y_s, y90, end_idx, post_end_idx, "down")
        t10 = crossing_time(t, y_s, y10, end_idx, post_end_idx, "down")
        decay_time = t10 - t90
        seg = (np.arange(len(t)) >= end_idx) & (np.arange(len(t)) < post_end_idx)
        corrected = np.clip(y_s[seg] - baseline, 0, None)
        area = float(np.trapz(corrected, t[seg])) if int(seg.sum()) > 2 else float("nan")
        if np.isfinite(decay_time) and decay_time > 0:
            decays.append(
                {
                    "edge_time": decay_t,
                    "decay_time": float(decay_time),
                    "decay_area": area,
                    "amplitude": float(amp),
                    "high_state": high_state,
                    "baseline": baseline,
                    "pre_noise": pre_noise,
                    "post_noise": post_noise,
                }
            )
    return rises, decays

def unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs = np.asarray(x[order], dtype=float)
    ys = np.asarray(y[order], dtype=float)
    xs, idx = np.unique(xs, return_index=True)
    ys = ys[idx]
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]
