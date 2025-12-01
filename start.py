# start.py
# Unified entry-point for VidSim.
# Provides both the Tkinter GUI launcher and a CLI interface to run the core pipeline with overrides.

import argparse
from typing import Any, Dict, Optional
from tkinter import messagebox

from ui import UI as ui
from _version import __version__
from core import main as core_main
from core.parameters import Parameters


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
        "high_quality_factor": Parameters.high_quality_factor,
        "low_quality_factor": Parameters.low_quality_factor,
        "gop_coefficient": Parameters.gop_coefficient,
        "roi_threshold": Parameters.roi_threshold,
        "threshold": Parameters.threshold,
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
        "high_quality_factor",
        "low_quality_factor",
        "gop_coefficient",
        "roi_threshold",
        "threshold",
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
    _verify_core_requirements()
    Parameters.setup_directories()
    core_main.run_coding()
    if args.decode:
        core_main.run_decoding()


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