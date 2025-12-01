"""Utilities to generate all standard VidSim plots from trace files."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from metrics import ploting


class PlotSpec:
    """Metadata describing how to invoke a plotting helper."""

    def __init__(
        self,
        name: str,
        func,
        sources: Sequence[str],
        requirements: Dict[str, Sequence[str]],
    ):
        self.name = name
        self.func = func
        self.sources = sources
        self.requirements = requirements


PLOT_SPECS: Tuple[PlotSpec, ...] = (
    PlotSpec("psnr", ploting.plot_psnr, ("rt_frame",), {"rt_frame": ("Rank", "PSNR")}),
    PlotSpec("ref_psnr", ploting.plot_ref_psnr, ("st_frame",), {"st_frame": ("Rank", "refPSNR")}),
    PlotSpec(
        "psnr_bpp",
        ploting.plot_psnr_bpp,
        ("rt_frame", "st_frame"),
        {"rt_frame": ("PSNR",), "st_frame": ("bpp",)},
    ),
    PlotSpec("ssim", ploting.plot_ssim, ("rt_frame",), {"rt_frame": ("Rank", "SSIM")}),
    PlotSpec(
        "ssim_bpp",
        ploting.plot_ssim_bpp,
        ("rt_frame", "st_frame"),
        {"rt_frame": ("SSIM",), "st_frame": ("bpp",)},
    ),
    PlotSpec("ref_ssim", ploting.plot_ref_ssim, ("st_frame",), {"st_frame": ("Rank", "refSSIM")}),
    PlotSpec(
        "brisque",
        ploting.plot_brisque,
        ("st_frame", "rt_frame"),
        {"st_frame": ("Rank", "refBRISQUE"), "rt_frame": ("BRISQUE", "Rank")},
    ),
    PlotSpec(
        "brisque_bpp",
        ploting.plot_brisque_bpp,
        ("rt_frame", "st_frame"),
        {"rt_frame": ("BRISQUE",), "st_frame": ("bpp",)},
    ),
    PlotSpec("bitrate", ploting.plot_bitrate, ("st_frame",), {"st_frame": ("Rank", "bit rate (kbps)")}),
    PlotSpec("bpp", ploting.plot_bpp, ("st_frame",), {"st_frame": ("Rank", "bpp")}),
    PlotSpec("encoding_energy", ploting.plot_energy, ("st_frame",), {"st_frame": ("Rank", "encodingEnergy(mJ)")}),
    PlotSpec("total_energy", ploting.plot_captured_energy, ("st_frame",), {"st_frame": ("Rank", "captureEnergy(mJ)", "encodingEnergy(mJ)")}),
    PlotSpec("frame_size", ploting.plot_data_size, ("st_frame",), {"st_frame": ("Rank", "Size(Bytes)")}),
    PlotSpec(
        "packet_size_over_time",
        ploting.plot_packet_size_over_time,
        ("packet",),
        {"packet": ("time", "pktSize")},
    ),
    PlotSpec("ber", ploting.plot_ber, ("packet",), {"packet": ("time", "BER")}),
    PlotSpec("snr", ploting.plot_snr, ("packet",), {"packet": ("time", "SNR")}),
    PlotSpec(
        "signal_loss",
        ploting.plot_signal_lost,
        ("packet",),
        {"packet": ("time", "signal_lost(db)")},
    ),
)


def _load_trace_file(path: str, rename_map: Dict[str, str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trace file: {path}")
    df = pd.read_csv(path, delimiter="\t")
    for source, target in rename_map.items():
        if source in df.columns:
            df.rename(columns={source: target}, inplace=True)
    return df


def _has_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    return all(col in df.columns for col in columns)


def _save_plot(spec: PlotSpec, data_map: Dict[str, pd.DataFrame], output_dir: str) -> None:
    # Validate required columns
    for source in spec.sources:
        requirements = spec.requirements.get(source, ())
        if not requirements:
            continue
        if not _has_required_columns(data_map[source], requirements):
            raise KeyError(f"Missing required columns {requirements} in {source}")

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        spec.func(ax, *[data_map[source] for source in spec.sources])
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{spec.name}.png"))
    finally:
        plt.close(fig)


def generate_all_plots(trace_dir: str, output_dir: str) -> None:
    """Load trace data and generate the standard VidSim plots."""
    os.makedirs(output_dir, exist_ok=True)
    st_frame_path = os.path.join(trace_dir, "st-frame")
    rt_frame_path = os.path.join(trace_dir, "rt-frame")
    packet_path = os.path.join(trace_dir, "st-packet")

    st_frame = _load_trace_file(st_frame_path, {"#Rank": "Rank", "refBrisque": "refBRISQUE"})
    rt_frame = _load_trace_file(rt_frame_path, {"#Rank": "Rank", "Brisque": "BRISQUE"})
    packet = _load_trace_file(packet_path, {"#time": "time"})

    data_map = {
        "st_frame": st_frame,
        "rt_frame": rt_frame,
        "packet": packet,
    }

    for spec in PLOT_SPECS:
        try:
            _save_plot(spec, data_map, output_dir)
        except (KeyError, ValueError) as exc:
            print(f"[plots] Skipped {spec.name}: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"[plots] Failed {spec.name}: {exc}")

