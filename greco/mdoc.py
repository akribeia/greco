"""Minimal MDOC parser for TITAN/FEI cryo-ET tilt series.

Returns the per-tilt records needed by the greco preprocessing chain.  The
only value the caller must supply externally is the total dose per tilt;
the parser also reports the measured per-frame doses so both can be used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re


_TILT_AXIS = re.compile(r"\bTilt\s*Axis\s*Angle\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", re.IGNORECASE)


def tilt_axis_from_line(line: str) -> float | None:
    """Read SerialEM's compact field or spaced title, including comma suffixes."""
    match = _TILT_AXIS.search(line)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


@dataclass
class TiltRecord:
    z_value: int
    angle_deg: float
    movie: str
    n_frames: int
    frame_dose_e_per_A2: float
    prior_dose_e_per_A2: float
    defocus_um: float | None
    rotation_angle_deg: float | None
    exposure_time_s: float | None
    counts_per_electron: float | None
    sub_frame_path: str | None = None


@dataclass
class Mdoc:
    voltage_kv: float
    pixel_size_A: float
    image_size: tuple[int, int]
    tilt_axis_angle_deg: float | None
    tilts: list[TiltRecord] = field(default_factory=list)


def parse_mdoc(path: Path) -> Mdoc:
    voltage = None
    pixel = None
    size = (0, 0)
    axis = None
    tilts: list[TiltRecord] = []
    current: dict | None = None

    def _kv(text: str):
        """Extract the first 'Key = value' pair from a mdoc fragment."""
        m = text.split("=")
        if len(m) < 2:
            return None, None
        return m[0].strip(), "=".join(m[1:]).strip()

    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        found_axis = tilt_axis_from_line(line)
        if found_axis is not None:
            axis = found_axis
        if line.startswith("[") and line.endswith("]"):
            inner = line[1:-1].strip()
            if "ZValue" in inner:
                _, zval = _kv(inner)
                if zval is not None and zval.isdigit():
                    if current is not None:
                        tilts.append(_finish(current))
                    current = {"z": int(zval)}
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(";", 1)[0].strip()

        if key in ("DataMode", "ImageFile"):
            continue
        if key == "Voltage":
            voltage = float(value)
        elif key == "PixelSpacing":
            pixel = float(value)
        elif key == "ImageSize":
            parts = value.split()
            size = (int(parts[0]), int(parts[1]))
        elif key == "ZValue":
            if current is not None:
                tilts.append(_finish(current))
            current = {"z": int(value)}
        elif current is not None:
            if key == "TiltAngle":
                current["angle"] = float(value)
            elif key == "NumSubFrames":
                current["n_frames"] = int(value)
            elif key == "SubFramePath":
                current["movie"] = value.split("\\")[-1]
                current["sub_frame_path"] = value
            elif key == "FrameDosesAndNumber":
                parts = value.split()
                current["frame_dose"] = float(parts[0])
                if len(parts) > 1 and "n_frames" not in current:
                    current["n_frames"] = int(parts[1])
            elif key == "PriorRecordDose":
                current["prior_dose"] = float(value)
            elif key == "Defocus":
                current["defocus"] = float(value)  # TITAN mdoc reports defocus in um
            elif key == "RotationAngle":
                current["rotation"] = float(value)
            elif key == "ExposureTime":
                current["exposure"] = float(value)
            elif key == "CountsPerElectron":
                current["cpe"] = float(value)

    if current is not None:
        tilts.append(_finish(current))
    if voltage is None or pixel is None or not tilts:
        raise ValueError(f"could not parse {path}: voltage={voltage} pixel={pixel} tilts={len(tilts)}")
    return Mdoc(voltage_kv=voltage, pixel_size_A=pixel, image_size=size,
                tilt_axis_angle_deg=axis, tilts=tilts)


def _finish(c: dict) -> TiltRecord:
    n_frames = c.get("n_frames", 0)
    frame_dose = c.get("frame_dose", 0.0)
    return TiltRecord(
        z_value=int(c["z"]),
        angle_deg=float(c.get("angle", 0.0)),
        movie=str(c.get("movie", "")),
        n_frames=n_frames,
        frame_dose_e_per_A2=frame_dose,
        prior_dose_e_per_A2=float(c.get("prior_dose", 0.0)),
        defocus_um=c.get("defocus"),
        rotation_angle_deg=c.get("rotation"),
        exposure_time_s=c.get("exposure"),
        counts_per_electron=c.get("cpe"),
        sub_frame_path=c.get("sub_frame_path"),
    )
