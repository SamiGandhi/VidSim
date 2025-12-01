# start.py
# Unified entry-point for VidSim.
# Provides both the Tkinter GUI launcher and a CLI interface to run the core pipeline with overrides.

import argparse
import os
from typing import Any, Dict
from tkinter import messagebox

from ui import UI as ui
from _version import __version__
from core import main as core_main
from core.parameters import Parameters
from metrics.plot_runner import generate_all_plots


def _get_parameter_defaults() -> Dict[str, Any]:
    """Expose current Parameter values for help text / defaults."""
    return {
        "captured_video_path": Parameters.captured_video_path,
        "output_directory": Parameters.output_directory,
        "method": Parameters.method or "Non-ROI",
        "fps": Parameters.fps,
        "default_width": Parameters.default_width,
        "default_height": Parameters.default_height,
        "quality_factor": Parameters.quality_factor,
        "zone_size": Parameters.zone_size,
        "DCT": Parameters.DCT,
        "level_numbers": Parameters.level_numbers,
        "entropy_coding": Parameters.entropy_coding,
        "high_quality_factor": Parameters.high_quality_factor,
        "low_quality_factor": Parameters.low_quality_factor,
        "gop_coefficient": Parameters.gop_coefficient,
        "roi_threshold": Parameters.roi_threshold,
        "threshold": Parameters.threshold,
        "w1": Parameters.w1,
        "w2": Parameters.w2,
        "w3": Parameters.w3,
        "max_level_S": Parameters.max_level_S,
        "DISTANCE": Parameters.DISTANCE,
        "FREQUENCY": Parameters.FREQUENCY,
        "ENVIRONMENT": Parameters.ENVIRONMENT,
        "HUMIDITY_LEVEL": Parameters.HUMIDITY_LEVEL,
        "VEGETATION_DENSITY_LEVEL": Parameters.VEGETATION_DENSITY_LEVEL,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    defaults = _get_parameter_defaults()

    parser = argparse.ArgumentParser(
        description="VidSim launcher. Use the GUI (default) or run the core pipeline via CLI arguments."
    )
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")
    parser.set_defaults(mode="ui")

    # UI mode (default)
    ui_parser = subparsers.add_parser("ui", help="Launch the Tkinter GUI (default)")
    ui_parser.add_argument("--version", action="store_true", help="Show VidSim version and exit.")

    # Core CLI mode
    core_parser = subparsers.add_parser("core", help="Run the core pipeline from the console")
    core_parser.add_argument(
        "--video-path",
        dest="captured_video_path",
        default=None,
        help=f"Input video file path (default: {defaults['captured_video_path'] or 'None'})",
    )
    core_parser.add_argument(
        "--output-dir",
        dest="output_directory",
        default=None,
        help=f"Override output base directory (default: {defaults['output_directory'] or 'auto'})",
    )
    core_parser.add_argument(
        "--method",
        choices=["ROI", "Non-ROI"],
        default=None,
        help=f"Encoding method (default: {defaults['method']})",
    )
    core_parser.add_argument(
        "--fps", type=int, default=None, help=f"Frames per second (default: {defaults['fps']})"
    )
    core_parser.add_argument(
        "--width", type=int, dest="default_width", default=None, help=f"Frame width (default: {defaults['default_width']})"
    )
    core_parser.add_argument(
        "--height",
        type=int,
        dest="default_height",
        default=None,
        help=f"Frame height (default: {defaults['default_height']})",
    )
    core_parser.add_argument(
        "--quality-factor",
        type=int,
        dest="quality_factor",
        default=None,
        help=f"Global quality factor (default: {defaults['quality_factor']})",
    )
    core_parser.add_argument(
        "--hqf",
        type=int,
        dest="high_quality_factor",
        default=None,
        help=f"ROI high quality factor (default: {defaults['high_quality_factor']})",
    )
    core_parser.add_argument(
        "--lqf",
        type=int,
        dest="low_quality_factor",
        default=None,
        help=f"ROI low quality factor (default: {defaults['low_quality_factor']})",
    )
    core_parser.add_argument(
        "--gop",
        type=int,
        dest="gop_coefficient",
        default=None,
        help=f"GOP coefficient / scene change threshold (default: {defaults['gop_coefficient']})",
    )
    core_parser.add_argument(
        "--roi-threshold",
        type=int,
        dest="roi_threshold",
        default=None,
        help=f"ROI block mean threshold (default: {defaults['roi_threshold']})",
    )
    core_parser.add_argument(
        "--threshold",
        type=int,
        dest="threshold",
        default=None,
        help=f"Binary threshold value (default: {defaults['threshold']})",
    )
    core_parser.add_argument(
        "--zone-size",
        type=int,
        dest="zone_size",
        default=None,
        help=f"DCT/ROI block size (default: {defaults['zone_size']})",
    )
    core_parser.add_argument(
        "--dct",
        dest="DCT",
        default=None,
        help=f"DCT type, e.g. CLA, sLLM, tBIN (default: {defaults['DCT']})",
    )
    core_parser.add_argument(
        "--levels",
        type=int,
        dest="level_numbers",
        default=None,
        help=f"Number of scalable layers (default: {defaults['level_numbers']})",
    )
    core_parser.add_argument(
        "--entropy",
        dest="entropy_coding",
        default=None,
        help=f"Entropy coding mode (default: {defaults['entropy_coding']})",
    )
    core_parser.add_argument(
        "--w1",
        type=int,
        dest="w1",
        default=None,
        help=f"ROI mask window size w1 (default: {defaults['w1']})",
    )
    core_parser.add_argument(
        "--w2",
        type=int,
        dest="w2",
        default=None,
        help=f"ROI mask window size w2 (default: {defaults['w2']})",
    )
    core_parser.add_argument(
        "--w3",
        type=int,
        dest="w3",
        default=None,
        help=f"ROI mask window size w3 (default: {defaults['w3']})",
    )
        
    core_parser.add_argument(
        "--max-level-s",
        type=int,
        dest="max_level_S",
        default=None,
        help=f"Max scalability level for S frames (default: {defaults['max_level_S']})",
    )
    core_parser.add_argument(
        "--distance",
        type=float,
        dest="DISTANCE",
        default=None,
        help=f"Transmission distance in meters (default: {defaults['DISTANCE']})",
    )
    core_parser.add_argument(
        "--frequency",
        type=float,
        dest="FREQUENCY",
        default=None,
        help=f"Carrier frequency in Hz (default: {defaults['FREQUENCY']})",
    )
    core_parser.add_argument(
        "--environment",
        dest="ENVIRONMENT",
        default=None,
        help=f"Environment profile (default: {defaults['ENVIRONMENT']})",
    )
    core_parser.add_argument(
        "--humidity",
        type=float,
        dest="HUMIDITY_LEVEL",
        default=None,
        help=f"Humidity percentage (default: {defaults['HUMIDITY_LEVEL']})",
    )
    core_parser.add_argument(
        "--vegetation",
        type=float,
        dest="VEGETATION_DENSITY_LEVEL",
        default=None,
        help=f"Vegetation density (default: {defaults['VEGETATION_DENSITY_LEVEL']})",
    )
    core_parser.add_argument(
        "--decode",
        action="store_true",
        help="Run decoding immediately after encoding.",
    )
    core_parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Skip encoding and only run decoding for an existing output directory.",
    )
    core_parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots based on trace data after processing.",
    )
    core_parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip encoding/decoding and only generate plots from an existing trace directory.",
    )
    core_parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory where generated plots will be saved (default: <trace>/plots).",
    )

    return parser


def _apply_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides back to the Parameters class."""
    override_fields = [
        "captured_video_path",
        "output_directory",
        "method",
        "fps",
        "default_width",
        "default_height",
        "quality_factor",
        "zone_size",
        "DCT",
        "level_numbers",
        "entropy_coding",
        "high_quality_factor",
        "low_quality_factor",
        "gop_coefficient",
        "roi_threshold",
        "threshold",
        "w1",
        "w2",
        "w3",
        "max_level_S",
        "DISTANCE",
        "FREQUENCY",
        "ENVIRONMENT",
        "HUMIDITY_LEVEL",
        "VEGETATION_DENSITY_LEVEL",
    ]

    for field in override_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(Parameters, field, value)


def _verify_core_requirements() -> None:
    """Ensure required parameters are set before running the core pipeline."""
    if not Parameters.captured_video_path:
        raise ValueError(
            "captured_video_path is required. Provide --video-path or set it in core/parameters.py."
        )


def start_ui(show_version: bool = False) -> None:
    """Launch the Tkinter UI."""
    print(f"Starting VidSim (Version: {__version__})")
    if show_version:
        print("VidSim is running in GUI mode.")

    root = ui.init_root_view()
    ui.open_root_view(root)


def run_core_cli(args: argparse.Namespace) -> None:
    """Execute the encoding/decoding pipeline via CLI mode."""
    _apply_overrides(args)

    decode_only = False
    plots_only = False
    if args.plots_only:
        args.plots = True
        plots_only = True
        args.decode = False
        args.decode_only = False

    if args.decode_only:
        args.decode = True
        decode_only = True
    elif args.decode and not Parameters.captured_video_path and Parameters.output_directory:
        decode_only = True

    if decode_only and not (Parameters.output_directory or Parameters.captured_video_path):
        raise ValueError("Decode-only mode requires --output-dir or --video-path to locate artifacts.")
    if plots_only and not (Parameters.output_directory or Parameters.captured_video_path):
        raise ValueError("Plots-only mode requires --output-dir or --video-path to locate artifacts.")

    should_encode = not decode_only and not plots_only
    if should_encode:
        _verify_core_requirements()

    Parameters.setup_directories()

    if should_encode:
        core_main.run_coding()

    if args.decode and not plots_only:
        core_main.run_decoding()

    if args.plots:
        plots_dir = args.plots_dir or os.path.join(Parameters.trace_file_path, "plots")
        generate_all_plots(Parameters.trace_file_path, plots_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "core":
        run_core_cli(args)
        return

    # Default to UI mode
    try:
        start_ui(show_version=getattr(args, "version", False))
    except Exception as exc:
        messagebox.showerror("VidSim Error", str(exc))
        start_ui(show_version=True)


if __name__ == "__main__":
    main()