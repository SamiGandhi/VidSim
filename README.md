# VidSim – Comprehensive Video Coding & Quality Analysis Suite

VidSim is an end-to-end toolkit for testing and evaluating video coding strategies with a special focus on Region-of-Interest (ROI) workflows. The project bundles:

- A configurable encoding/decoding pipeline (standard & ROI-aware).
- Detailed quality and energy metrics.
- Automated plotting and reporting.
- A Tkinter-based user interface.
- Command-line automation for headless runs.

VidSim is built for researchers and engineers who need to iterate quickly on compression experiments, visualize results, and share repeatable scenarios.

---

## Table of Contents

1. [Key Highlights](#key-highlights)
2. [Architecture Overview](#architecture-overview)
3. [Directory Layout](#directory-layout)
4. [Installation & Environment](#installation--environment)
5. [Entry Points](#entry-points)
6. [Graphical Interface](#graphical-interface)
7. [Command-Line Interface](#command-line-interface)
8. [Configuration & Parameters](#configuration--parameters)
9. [Data Flow & Outputs](#data-flow--outputs)
10. [Quality Metrics & Plotting](#quality-metrics--plotting)
11. [Petri Net & Analytical Tools](#petri-net--analytical-tools)
12. [Advanced Topics](#advanced-topics)
13. [Troubleshooting & FAQ](#troubleshooting--faq)
14. [Contributing](#contributing)
15. [Contacts](#contacts)
16. [License](#license)

---

## Key Highlights

### Core Video Pipeline
- **ROI & Non-ROI Encoding**: Encode full frames or restrict bitrate to ROI masks.
- **Adaptive GOP**: Detect scene changes via MSE thresholds.
- **Main vs Secondary Frames**: Encode intra and inter frames separately.
- **Scalable Layers**: Configure up to multiple layers with entropy coding.

### Quality & Analysis
- **Metrics**: PSNR, SSIM, BRISQUE (no-reference), bitrates, packet stats.
- **Plots**: Energy curves, rate-distortion, BER/SNR over time, ROI vs non-ROI comparisons.
- **Trace Files**: Frame-level and packet-level CSV/TSV logs for offline processing.

### Tooling
- **Tkinter GUI**: Parameter setup, run control, visualization.
- **CLI Automation**: Headless runs, decode-only, plots-only, directory-based workflows.
- **Plot Runner**: One command to build all graph outputs for an experiment.
- **Packaging**: PyInstaller-ready structure for Windows distribution.

---

## Architecture Overview

```
┌──────────────────┐
│  GUI / CLI       │  (start.py / vidsim.py)
└────────┬─────────┘
         │ Parameters (core/parameters.py)
         ▼
┌──────────────────┐
│  Core Pipeline   │  (capture → ROI/main encoders → entropy → trace)
└────────┬─────────┘
         │ Trace files (st-frame / st-packet / rt-frame)
         ▼
┌──────────────────┐
│ Decoder & Stats  │  (core/decoder.py, metrics/quality_metrics.py)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Plot Runner      │  (metrics/plot_runner.py → PNG dashboards)
└──────────────────┘
```

- **Input**: Any video supported by OpenCV (e.g., MP4, AVI, Y4M).
- **Processing**: ROI detection, DCT transform, quantization, scalable layer packaging.
- **Output**: Captured frames, reference frames, decoded frames, trace logs, metric plots.

---

## Directory Layout

```
vidsim_/
├── core/                  # Capture, encoding, decoding, utilities
├── metrics/               # Quality metrics & plotting helpers
├── models/                # Petri net / analytical models
├── ui/                    # Tkinter front-end
├── res/                   # Icons & static assets
├── start.py               # Main entry point (GUI + CLI)
├── vidsim.py              # Thin wrapper around start.py (canonical name)
├── metrics/plot_runner.py # Automated plot orchestrator
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

Experiment outputs follow the pattern:

```
<base>/<height>x<width>/<PROFILE>/GOP_<value>/
├── captured_frames/
├── reference_frames/
├── decoded/
├── roi_frames/ (ROI mode)
├── roi_masks/  (ROI mode)
└── trace/
    ├── st-frame     (encoding stats)
    ├── st-packet    (packet logs)
    └── rt-frame     (decoding metrics)
```

If you pass `--output-dir`, `<base>` becomes that directory; otherwise it is derived from the source video path.

---

## Installation & Environment

1. Clone and enter the project:
   ```bash
   git clone https://github.com/SamiGandhi/VidSim
   cd vidsim_
   ```

2. Create & activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Smoke test:
   ```bash
   python vidsim.py --help
   python vidsim.py core --help
   ```

> **Note**: Tkinter ships with CPython, but on Linux you might need `sudo apt install python3-tk`. BRISQUE relies on libsvm; the bundled `metrics/brisque/` folder must remain in place.

---

## Entry Points

| Command                      | Description                                                       |
|------------------------------|-------------------------------------------------------------------|
| `python vidsim.py`           | Launch GUI (default).                                             |
| `python start.py`            | Same as above (alternate name).                                  |
| `python vidsim.py ui`        | Explicit GUI mode + `--version` flag to print only the version.  |
| `python vidsim.py core ...`  | CLI automation (see below).                                      |

`vidsim.py` simply imports `start.main()` to align with packaging conventions. If you bundle with PyInstaller, target `vidsim.py`.

---

## Graphical Interface

Run:
```bash
python vidsim.py
```

Features:
- File pickers for video selection & output directories.
- Parameter panels (encoding, ROI, network, energy).
- Buttons for *Capture*, *Encode*, *Decode*, *Compute Metrics*, *Generate Plots*.
- Embedded Matplotlib widgets (via `FigureCanvasTkAgg`) for inline visualization.
- Status console using `ScrolledText` for logs.

The GUI writes chosen parameters back to `core/parameters.py` at runtime, so CLI runs can reuse them.

---

## Command-Line Interface

### Core Workflow

```bash
python vidsim.py core \
    --video-path input.mp4 \
    --method ROI \
    --quality-factor 75 \
    --roi-threshold 12 \
    --decode \
    --plots
```

Steps performed:
1. Apply overrides (`--video-path`, `--method`, …) on top of `Parameters`.
2. Create output directories.
3. Capture frames, encode, write traces.
4. Decode (if `--decode`).
5. Generate plots (if `--plots`).

### Decode-Only

If you already have captured frames / traces:

```bash
python vidsim.py core \
    --output-dir vir \
    --decode-only
```

The CLI automatically counts existing frames if `--video-path` is omitted.

### Plot-Only

Generate dashboards for a previous experiment:

```bash
python vidsim.py core \
    --output-dir vir \
    --plots-only \
    --plots-dir vir/plots
```

### CLI Options Cheat Sheet

| Flag | Description |
|------|-------------|
| `--video-path PATH` | Input video; required for new encodes. |
| `--output-dir DIR` | Root directory for results (capture, trace, decoded, plots). |
| `--method {ROI,Non-ROI}` | Encoding strategy. |
| `--fps`, `--width`, `--height` | Capture rate & resizing. Floats allowed for FPS. |
| `--quality-factor`, `--hqf`, `--lqf` | JPEG/ROI quality controls. |
| `--zone-size`, `--dct`, `--levels`, `--entropy` | Transform & scalability settings. |
| `--gop` | GOP/MSE threshold for main-vs-secondary frame decision. |
| `--roi-threshold`, `--threshold`, `--w1/2/3`, `--max-level-s` | ROI mask tuning. |
| `--distance`, `--frequency`, `--environment`, `--humidity`, `--vegetation` | Network model overrides. |
| `--decode`, `--decode-only` | Run decoder; optionally skip encoding. |
| `--plots`, `--plots-only`, `--plots-dir` | Plot generation controls. |

Run `python vidsim.py core --help` for the authoritative list (kept in sync with `start.py`).

---

## Configuration & Parameters

All defaults live in `core/parameters.py`. Highlights:

| Group | Key Attributes |
|-------|----------------|
| Video | `captured_video_path`, `fps`, `default_width`, `default_height` |
| Encoding | `method`, `quality_factor`, `high_quality_factor`, `low_quality_factor`, `gop_coefficient`, `zone_size`, `DCT`, `level_numbers`, `entropy_coding` |
| ROI | `w1`, `w2`, `w3`, `roi_threshold`, `threshold`, `max_level_S` |
| Paths | `output_directory`, derived directories via `setup_directories()` |
| Network/Energy | `DISTANCE`, `FREQUENCY`, `ENVIRONMENT`, `HUMIDITY_LEVEL`, `VEGETATION_DENSITY_LEVEL`, `POWER`, `PROC_CLOCK`, `CAPTURE_E_PER_BLOCK` |

### Tips
- Call `Parameters.setup_directories()` after setting all fields; it derives folder paths.
- Use `Parameters.print_params()` inside Python or REPL to inspect the effective configuration.
- `Parameters.reset()` restores shipping defaults.

---

## Data Flow & Outputs

1. **Capture**: `core/capture.py` reads every `capture_steps` frame, converts to grayscale, resizes.
2. **ROI Detection**: `core/roi.py` produces SAD/R/G maps, compression classes, masks saved in `roi_masks/`.
3. **Encoding**:
   - DCT/quantization via `core/main_frame.py` & `core/dct.py`.
   - Scalable layers with `core/entropy.py`.
   - Packetization recorded in `core/trace.py` structures and written by `core/util.py`.
4. **Decoding**:
   - Reads `st-packet`, reconstructs frames in `core/decoder.py`.
   - BER/SNR optional noise injection.
5. **Metrics**:
   - `metrics/quality_metrics.py` computes PSNR/SSIM/BRISQUE (ROI vs original).
6. **Plots**:
   - `metrics/plot_runner.generate_all_plots()` loads trace TSVs and saves PNGs.

---

## Quality Metrics & Plotting

### Metrics

- **PSNR**: `calculate_psnr(ref, decoded)` – uses `skimage.metrics.peak_signal_noise_ratio`.
- **SSIM**: `calculate_ssim(ref, decoded)` – window size 8 by default.
- **BRISQUE**: `calculate_brisque(rgb_image)` – no-reference; requires BRISQUE folder.

### Plot Runner Outputs

When `--plots` or `--plots-only` is provided, the following PNGs are produced (default: `<trace>/plots/`):

| File | Description |
|------|-------------|
| `psnr.png`, `ssim.png`, `brisque.png` | Metric vs frame index. |
| `psnr_bpp.png`, `ssim_bpp.png`, `brisque_bpp.png` | Rate-distortion views. |
| `bitrate.png`, `bpp.png`, `frame_size.png` | Size/bandwidth tracking. |
| `encoding_energy.png`, `total_energy.png` | Energy breakdowns. |
| `packet_size_over_time.png`, `ber.png`, `snr.png`, `signal_loss.png` | Network/packet diagnostics. |
| `ref_psnr.png`, `ref_ssim.png` | Reference vs decoded comparisons. |

`plot_runner` automatically renames columns (e.g., `#Rank → Rank`, `refBrisque → refBRISQUE`) to handle trace inconsistencies.

---

## Petri Net & Analytical Tools

Run the Petri net visualizer to understand the end-to-end workflow:

```bash
python -m models.perti_net
```

It uses `networkx` + Matplotlib to draw the discrete-event model (places, transitions, tokens). Useful for presentations and debugging the states (capture → encode → packetize → send → decode → reconstruct).

---

## Advanced Topics

### Fractional FPS
- The CLI accepts floats (`--fps 29.97`).
- Internally we store `Parameters.fps` as-is; ensure any arithmetic handles floats.
- GUI currently writes integers; adjust UI code if you need fractional capture rates from the UI.

### Packaging (PyInstaller)

Examples:

```bash
# directory bundle (easier to debug)
pyinstaller --name vidsim --onedir vidsim.py

# single executable
pyinstaller --name vidsim --onefile vidsim.py \
  --add-data "ui;ui" --add-data "core;core" --add-data "metrics;metrics" --add-data "res;res" \
  --icon=res/icon.png
```

Guidelines:
- Build on the target architecture (x86 vs x64).
- Remember VC++ Redistributable requirements for OpenCV.
- Use `resource_path()` helpers in code when opening files so PyInstaller’s `_MEIPASS` is respected.

```python
def resource_path(rel_path):
    import os, sys
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)
```

### Persistent Configurations
- Save parameter presets as JSON/YAML and load them before running `start.main()` if needed.
- For multiple experiments, script over `python vidsim.py core --output-dir run_<N> ...`.

---

## Troubleshooting & FAQ

| Issue | Cause / Fix |
|-------|-------------|
| `captured_video_path is required` | Provide `--video-path` or ensure `core/parameters.py` has one. For decode-only, add `--output-dir`. |
| GUI doesn’t launch | Ensure Tkinter is available (`python -m tkinter`). On Linux install the Tk package. |
| BRISQUE errors | Keep `metrics/brisque/` intact; run from root folder so relative imports work. |
| No plots generated | Check `trace/` files exist. `plot_runner` logs skipped plots with reasons (missing columns). |
| FPS parsing errors | Use float-friendly CLI (`--fps 29.97`) or adjust UI to accept decimals. |
| Permission denied writing outputs | Run from a location where you have write access, or specify `--output-dir`. |
| Antivirus flags PyInstaller EXE | Use `--onedir` during testing; sign binaries for distribution. |

**FAQ**

- *Can I feed color videos?* Yes—frames are converted to grayscale for processing, but you can modify `capture.py` to keep RGB if needed.
- *Can I limit ROI to multiple regions?* Current implementation uses nested W1/W2/W3 windows; extend `roi.py` to add more classes.
- *How do I compare ROI vs non-ROI quickly?* Run two experiments with different `--method`, then compare plots or load traces in Jupyter.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-idea`.
3. Follow PEP 8, add docstrings/comments.
4. Update tests/plots/docs when relevant.
5. Submit a pull request with a clear description.

Please run linting/tests (`pytest`) before submitting. For major changes (e.g., new metrics), file an issue to discuss design choices.

---

## Contacts

- **HADJI Oussama** – University of Batna 2, Department of Computer Science, Constantine Road, Fésdis 05078, Batna, Algeria.  
  Email: ou.hadji@univ-batna2.dz

- **MAIMOUR Moufida** – Université de Lorraine, CNRS, CRAN, F-54000 Nancy, France.  
  Email: moufida.maimour@univ-lorraine.fr

For general questions, open an issue on GitHub. For private collaboration or academic inquiries, reach out via the above emails.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the exact terms.

---

**Last Updated:** 30-11-2025  
**Maintained by:** HADJI Oussama & MAIMOUR Moufida




---

## Test Video

The `TESTING_VIDEOS/` folder contains sample video file for quick testing and validation of the VidSim pipeline.

### Available Test Video
Selected From "A Large-scale Benchmark Dataset for Event Recognition in Surveillance Video" By 

| File | Resolution | FPS | Use Case |
|------|------------|-----|----------|
| `VIRAT_S_010204_05_000856_000890.mp4` | 144×144 | 30 | Quick smoke tests; fastest encoding. |


### Quick Start with Test Videos

Encode a test video and generate plots:

```bash
python vidsim.py core \
    --video-path "TESTING_VIDEOS\VIRAT_S_010204_05_000856_000890.mp4" \
    --output-dir "TESTING_VIDEOS\vir" \
    --method ROI \
    --quality-factor 75 \
    --decode \
    --plots
```

Decode-only (reusing previous captures):

```bash
python vidsim.py core \
    --output-dir "TESTING_VIDEOS\vir" \
    --decode-only
```

Generate plots for an existing experiment:

```bash
python vidsim.py core \
    --output-dir "vir\144X144\QF_75" \
    --plots-only
```

---

