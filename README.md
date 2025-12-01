# VidSim - Video Coding & Quality Analysis Framework

VidSim is an advanced research framework for region-of-interest (ROI) video coding, quality assessment, and visualization. The system provides both a GUI interface for video simulation experiments, supporting flexible encoding strategies, comprehensive quality metrics, and detailed analysis tools.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Quality Metrics](#quality-metrics)
- [Examples](#examples)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

- **Flexible Video Coding Pipeline**: Supports both standard and ROI-based encoding with configurable parameters
- **Region-of-Interest (ROI) Support**: Intelligent mask generation and block prioritization for adaptive compression
- **Scalable Layer Encoding**: Multi-layer encoding with configurable quality levels
- **Multiple DCT Algorithms**: Support for various DCT implementations (CLA, LLM, BIN variants)
- **Frame Type Management**: Main (M) and Secondary (S) frame encoding with GOP-based scene change detection

### Quality Assessment

- **Comprehensive Metrics**: PSNR, SSIM, and BRISQUE quality measurements
- **Automated Analysis**: Built-in tools for comparing ROI vs. non-ROI encoding strategies
- **Visualization Tools**: Extensive plotting capabilities for metrics, energy, bitrate, and network loss

### Analysis & Visualization

- **Energy Analysis**: Capture and encoding energy calculations with detailed breakdowns
- **Network Simulation**: Signal loss, SNR, and BER modeling for transmission analysis
- **Petri Net Modeling**: System workflow visualization and analysis
- **Trace File Generation**: Detailed logging of frames, packets, and metrics

### User Interface

- **Graphical User Interface**: Intuitive Tkinter-based GUI for easy experimentation
- **Batch Processing**: Command-line interface for automated runs
- **Real-time Monitoring**: Progress tracking and result visualization

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/SamiGandhi/VidSim
   cd vidsim_
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**
   ```bash
   python start.py --help
   ```

---

## Quick Start

### Running the GUI

Launch the graphical interface (default):

```bash
python start.py
```

This opens the Tkinter GUI where you can:
- Select video files
- Configure encoding parameters
- Run encoding/decoding pipelines
- Visualize results and metrics

Note: a small wrapper entrypoint was added so you can also run the project using the canonical entrypoint `vidsim.py`:

```bash
python vidsim.py      # same behavior as start.py
python vidsim.py --help
python vidsim.py core --help   # show all core CLI options (detailed)
```

### Running from the Console (CLI Mode)

You can run the core pipeline directly from the terminal and override any parameter. Only the values you supply are changed; everything else uses the defaults in `core/parameters.py`.

```bash
python start.py core \
    --video-path path/to/video.mp4 \
    --method ROI \
    --quality-factor 75 \
    --roi-threshold 12 \
    --decode
```

**Useful CLI options**

| Option | Description |
| --- | --- |
| `--video-path` | Path to the input video file (required if not set in `Parameters`) |
| `--output-dir` | Override the output base directory |
| `--method` | `ROI` or `Non-ROI` |
| `--fps`, `--width`, `--height` | Frame rate and resolution |
| `--quality-factor`, `--hqf`, `--lqf` | Quality settings |
| `--gop` | GOP coefficient / scene change threshold |
| `--roi-threshold`, `--threshold`, `--max-level-s` | ROI / S-frame controls |
| `--distance`, `--frequency`, `--environment`, `--humidity`, `--vegetation` | Channel model overrides |
| `--decode` | Run decoding immediately after encoding |
| `--decode-only` | Skip encoding and decode an existing run (requires `--output-dir` or `--video-path`) |

Run `python start.py core --help` to see the full list.

### Example Workflow

1. **Configure Parameters**: Edit `core/parameters.py`, use the GUI, or pass CLI flags to set:
   - Video input path
   - Encoding method (ROI or standard)
   - Quality factors
   - Frame dimensions
   - GOP settings

2. **Run Encoding**: Execute via GUI or CLI.

3. **Analyze Results**: Use plotting scripts in `metrics/` to visualize:
   - Quality metrics over time
   - Energy consumption
   - Bitrate analysis
   - Network loss characteristics

---

## Usage

### Core Pipeline

The main video processing pipeline is located in `core/`:

- **`main.py`**: Entry point for the coding pipeline
  - `run_coding()`: Encodes video frames
  - `run_decoding()`: Decodes and reconstructs video

- **`capture.py`**: Frame extraction from video files
  - Supports frame rate downsampling
  - Automatic resizing to target dimensions

- **`main_frame.py`**: Main frame encoding logic
  - DCT transformation
  - Quantization
  - Layer creation and packetization

- **`second_frame.py`**: Secondary frame encoding
  - Difference-based encoding
  - Block prioritization by Mean Square (MS) values

- **`roi.py`**: Region-of-Interest processing
  - SAD, R, and G mask generation
  - Compression class assignment (C1, C2, C3)

- **`decoder.py`**: Decoding and reconstruction
  - Frame reconstruction from packets
  - Error concealment
  - BER noise simulation

### Quality Metrics

Located in `metrics/quality_metrics.py`:

```python
from metrics import quality_metrics

# Calculate PSNR
psnr = quality_metrics.calculate_psnr(reference_image, decoded_image)

# Calculate SSIM
ssim = quality_metrics.calculate_ssim(reference_image, decoded_image)

# Calculate BRISQUE (requires 3-channel RGB image)
brisque_score = quality_metrics.calculate_brisque(rgb_image)
```

### Plotting Tools

The `metrics/` directory contains several plotting utilities:

- **`ploting.py`**: Comprehensive plotting functions for:
  - PSNR/SSIM/BRISQUE over frame sequence
  - Rate-distortion curves (bpp vs. quality)
  - Bitrate analysis
  - Energy consumption

- **`plot_energy.py`**: Energy comparison between ROI and non-ROI methods

- **`ploat_data_loss.py`**: Network loss visualization (signal loss, SNR, BER)

- **`plots_.py`**: Side-by-side comparison plots for different encoding strategies

### Petri Net Modeling

Visualize system workflow using the Petri net model:

```bash
python -m models.perti_net
```

This generates a graph showing the video processing pipeline stages.

---

## Project Structure

```
vidsim_/
├── core/                    # Core video processing pipeline
│   ├── main.py             # Main pipeline entry point
│   ├── capture.py          # Frame capture from video
│   ├── decoder.py          # Decoding and reconstruction
│   ├── main_frame.py       # Main frame encoding
│   ├── second_frame.py     # Secondary frame encoding
│   ├── second_frame_.py    # Alternative S-frame implementation
│   ├── roi.py              # Region-of-Interest processing
│   ├── util.py             # Utility functions
│   ├── parameters.py       # Configuration parameters
│   ├── trace.py            # Data structures for logging
│   ├── dct.py              # DCT transformation
│   ├── entropy.py          # Entropy coding
│   ├── network_losses_model.py  # Network loss modeling
│   └── db_losses.py        # Database loss utilities
│
├── metrics/                 # Quality metrics and plotting
│   ├── quality_metrics.py   # PSNR, SSIM, BRISQUE calculations
│   ├── ploting.py           # Main plotting functions
│   ├── plot_energy.py       # Energy comparison plots
│   ├── ploat_data_loss.py   # Network loss plots
│   ├── plots_.py            # Comparison plots
|   └── brisque/             # BRISQUE quality metric (third-party)
│         ├── brisque.py          # BRISQUE implementation
│         ├── models/             # Pre-trained models
│         └── utilities.py        # Helper functions
│
├── models/                  # Analytical models
│   └── perti_net.py        # Petri net system model
│
├── ui/                      # User interface
│   ├── UI.py               # Main GUI application
│   └── UI2.py              # Alternative UI implementation
├── res/                     # Resource files
│   ├── icon.png            # Application icon
│   └── *.png               # UI icons
│
├── start.py                 # Project entry point
├── _version.py              # Project version name
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Configuration

### Parameters File

All configuration is centralized in `core/parameters.py`. Key parameters include:

#### Video Settings
- `captured_video_path`: Input video file path
- `fps`: Target frames per second
- `default_width`, `default_height`: Frame dimensions

#### Encoding Settings
- `method`: Encoding method (`'ROI'` or standard)
- `quality_factor`: JPEG quality (1-100)
- `high_quality_factor`: ROI high-priority quality
- `low_quality_factor`: ROI low-priority quality
- `gop_coefficient`: Scene change threshold (MSE)
- `DCT`: DCT algorithm (`"CLA"`, `"sLLM"`, `"tBIN"`, etc.)
- `entropy_coding`: Entropy coding method (`"RLE_EG"`)

#### ROI Settings
- `w1`, `w2`, `w3`: Mask window sizes
- `roi_threshold`: Block mean threshold for region detection
- `threshold`: Binary thresholding value

#### Network Simulation
- `DISTANCE`: Transmission distance (meters)
- `FREQUENCY`: Carrier frequency (Hz)
- `ENVIRONMENT`: Environment type (`"wetland"`, etc.)
- `HUMIDITY_LEVEL`: Humidity percentage
- `VEGETATION_DENSITY_LEVEL`: Vegetation density

#### Energy Estimation
- `POWER`: Device power consumption
- `PROC_CLOCK`: Processor clock rate
- `CAPTURE_E_PER_BLOCK`: Energy per captured block

### Setting Up Parameters

**Option 1: Edit `core/parameters.py` directly**
```python
from core.parameters import Parameters

Parameters.captured_video_path = "path/to/video.mp4"
Parameters.method = "ROI"
Parameters.quality_factor = 80
Parameters.setup_directories()
```

**Option 2: Use the GUI**
The GUI provides a user-friendly interface to set all parameters without editing code.

---

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-level differences between reference and decoded images:
- Higher values indicate better quality
- Typical range: 20-50 dB
- Formula: `PSNR = 10 * log10(MAX² / MSE)`

### SSIM (Structural Similarity Index)

Perceptual quality metric considering luminance, contrast, and structure:
- Range: 0 to 1 (1 = perfect match)
- More aligned with human perception than PSNR

### BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

No-reference quality metric:
- Lower scores indicate better quality
- Works on RGB images (3 channels)
- Uses pre-trained SVM models

---

## Examples

### Example 1: Basic Encoding

```python
from core.parameters import Parameters
from core import main

# Configure parameters
Parameters.captured_video_path = "input_video.mp4"
Parameters.method = ""  # Standard encoding
Parameters.quality_factor = 80
Parameters.fps = 2
Parameters.setup_directories()

# Run encoding
main.run_coding()
main.run_decoding()
```

### Example 2: ROI Encoding

```python
from core.parameters import Parameters
from core import main

# Configure for ROI
Parameters.captured_video_path = "input_video.mp4"
Parameters.method = "ROI"
Parameters.high_quality_factor = 40
Parameters.low_quality_factor = 20
Parameters.roi_threshold = 10
Parameters.setup_directories()

# Run ROI encoding
main.run_coding()
main.run_decoding()
```

### Example 3: Quality Analysis

```python
import pandas as pd
from metrics import ploting
import matplotlib.pyplot as plt

# Load trace data
st_frame = pd.read_csv("trace/st-frame", delimiter="\t")
rt_frame = pd.read_csv("trace/rt-frame", delimiter="\t")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ploting.plot_psnr(axes[0, 0], rt_frame)
ploting.plot_ssim(axes[0, 1], rt_frame)
ploting.plot_brisque(axes[1, 0], st_frame, rt_frame)
ploting.plot_energy(axes[1, 1], st_frame)
plt.tight_layout()
plt.show()
```

---

## Requirements

All dependencies are listed in `requirements.txt`. Install via:

```bash
pip install -r requirements.txt
```

Key packages include:

### Core Dependencies
- **numpy**: Numerical computations
- **opencv-python**: Image/video processing
- **matplotlib**: Plotting and visualization
- **pandas**: Data analysis and CSV handling
- **scikit-image**: Image processing algorithms
- **scipy**: Scientific computing

### Quality Metrics
- **libsvm**: Support Vector Machine library

### GUI
- **tkinter**: GUI framework (built-in with Python)
- **Pmw**: Advanced Tkinter widgets

### Analysis
- **networkx**: Graph/networks analysis (for Petri nets)

### Development
- **pytest**: Testing framework

See `requirements.txt` for complete list with versions.

---

## Troubleshooting

### Common Issues

**1. Import Errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the project root directory
- Verify Python version (3.9+)

**2. Video File Not Found**
- Check `Parameters.captured_video_path` is set correctly
- Use absolute paths if relative paths fail
- Ensure video format is supported (MP4, AVI, etc.)

**3. GUI Not Launching**
- Verify tkinter is available: `python -m tkinter`
- On Linux, may need: `sudo apt-get install python3-tk`

**4. BRISQUE Errors**
- Ensure `brisque/` directory is present with all files
- Check that `brisque/models/` contains required model files

**5. Memory Issues**
- Reduce frame dimensions in `parameters.py`
- Process videos in smaller chunks
- Close other applications

**6. Permission Errors**
- Ensure write permissions for output directories
- Check that output paths are valid

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** with clear comments
4. **Test thoroughly** before submitting
5. **Submit a pull request** with a clear description

### Code Style
- Follow PEP 8 Python style guide
- Add docstrings to all functions and classes
- Include comments for complex logic
- Update documentation as needed

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact & Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Email**: (ou.hadji@univ-batna2.dz) or (moufida.maimour@univ-lorraine.fr) 
- **Documentation**: See inline code comments and docstrings

---

## Acknowledgments

- BRISQUE implementation and models (credit original authors)
- OpenCV community for excellent image processing tools
- All contributors and users of this framework

---

## Version History

- **v1.0.0**: Initial release with core functionality
  - ROI and standard encoding
  - Quality metrics (PSNR, SSIM, BRISQUE)
  - GUI interface
  - Comprehensive plotting tools

---

**Last Updated**: 30-11-2025

**Maintained by**: 
   - HADJI Oussama - University of Batna 2, Departement of Computer Science, Constantine Route. Fésdis, 05078 (Batna, Algeria) - ou.hadji@univ-batna2.dz
   - MAIMOUR Moufida - Université de Lorraine, CNRS, CRAN, F-54000 (Nancy, France) - moufida.maimour@univ-lorraine.fr

### Notes about FPS and numeric CLI args
- The CLI `--fps` historically expected an integer; fractional frame rates (e.g. 29.97) can appear from some video files.
- The UI writes the integer part of the captured FPS to avoid int("29.97") errors.
- Recommended: pass floats on the CLI and accept floats in the code. Example usage:

```bash
python vidsim.py core --video-path "C:\Videos\sample.mp4" --fps 29.97
```

If you prefer integers, use the integer part (e.g. `--fps 29`) or change the parser in `start.py` to accept floats (`type=float`) and ensure `Parameters.fps` can hold a float.

## Packaging (Windows executable)

You can create a Windows executable using PyInstaller. Use `--onedir` while debugging and switch to `--onefile` for distribution. Example commands:

```bash
# debug-friendly folder
pyinstaller --name vidsim --onedir start.py

# single-file executable (may be larger and extract at runtime)
pyinstaller --name vidsim --onefile start.py

# include package data (Windows uses semicolon separator)
pyinstaller --onefile vidsim.py --add-data "ui;ui" --add-data "core;core" --add-data "metrics;metrics" --add-data "res;res" --icon=res/icon.png
```

Common portability notes:
- Build for the target architecture (x64 vs x86).
- Ship or instruct users to install the correct Microsoft Visual C++ Redistributable.
- Use `--onedir` if PyInstaller misses large native DLLs (OpenCV); it's easier to debug.
- Antivirus/SmartScreen may block unsigned single-file EXEs — consider code signing or an installer.

Runtime helper to locate bundled resources (use when opening data files in code):

```python
def resource_path(rel_path):
    import sys, os
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)
```

For complex packaging, generate and edit a `.spec` file to include datas and hidden imports. Test the packaged app on a clean Windows VM before distributing.

