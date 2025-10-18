# Tinnitus Sound Therapy


This is a Python port of the MatLab code for generating audio stimulus files for tinnitus sound therapy based on the [research by Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., and Sedley, W.](https://www.sciencedirect.com/science/article/pii/S0378595525001534)

## Original Research

- **Authors**: Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
- **DOI**: https://doi.org/10.25405/data.ncl.27109693
- **License**: CC BY-NC-SA 4.0
- **Last updated**: 15/10/2024

## Overview

The Python programs here generate two kind of audio files:
* `hearing_assessment_stimuli.py` generates files intended for assessment of the tinnitus frequency and hearing loss slope. It reproduces the functionality of the original `Hearing_Tin_Stim_Generation.m` MatLab file.
* `tinnitus_sound_therapy.py` generates the therapeutic stimuli, as the original `Generate_Full_Experiment_Stimuli` and it can either generate the full set of files or specific files for given tinnitus frequency, hearing loss, etc.


## Installation

This project supports both traditional pip and modern uv package management.

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that automatically handles virtual environments.

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip
   pip install uv
   ```

2. **Install dependencies and run**:
   ```bash
   # uv automatically creates a virtual environment and installs dependencies
   uv run python tinnitus_sound_therapy.py --help
   uv run python hearing_assessment_stimuli.py --help
   ```

3. **For development** (installs in editable mode):
   ```bash
   uv sync
   uv run python test_implementation.py
   ```

### Option 2: Using pip with virtual environment

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the programs**:
   ```bash
   python tinnitus_sound_therapy.py --help
   python hearing_assessment_stimuli.py --help
   ```

### Required packages
- numpy>=1.21.0
- soundfile>=0.10.0
- scipy>=1.7.0

### Project files

- `tinnitus_sound_therapy.py` - Main CLI tool for generating therapeutic stimuli
- `hearing_assessment_stimuli.py` - CLI tool for hearing assessment stimuli generation
- `test_implementation.py` - Test suite to verify functionality
- `requirements.txt` - Python dependencies for pip
- `pyproject.toml` - Modern Python project configuration for uv

## Usage

This project provides two CLI tools for generating different types of audio stimuli for tinnitus research and therapy.

### 1. Therapeutic Stimuli Generation (`tinnitus_sound_therapy.py`)

This is the main tool for generating therapeutic audio stimuli with various modulation types.

#### Basic Usage

```bash
# Generate stimuli for specific tinnitus frequency (recommended)
python tinnitus_sound_therapy.py -f 3000

# Using uv
uv run python tinnitus_sound_therapy.py -f 3000
```

#### Complete CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--all` | Generate all experimental stimuli (84 files, ~60GB) | - |
| `-f, --frequency` | Tinnitus frequency in Hz (determines frequency band) | - |
| `--prefer-lower` | When frequency is on band boundary, prefer lower band | False (prefer higher) |
| `-m, --modulation` | Modulation type(s): `noise`, `amp`, `phase` | All types |
| `--hearing-profile` | Hearing loss profile(s): `NH`, `MildHL`, `ModHL`, `SevHL` | NH only |
| `-n, --files-per-category` | Number of files per category | 1 |
| `-o, --output-dir` | Output directory | Current directory |
| `--prefix` | Prefix for generated filenames | None |
| `--random-names` | Use random filenames instead of descriptive names | False |
| `--loud` | Loudness scaling factor | 0.1 |
| `-d, --duration` | Duration of each file in seconds | 3600 (1 hour) |

#### Frequency Band Mapping

The tool automatically maps tinnitus frequencies to appropriate frequency band pairs:

| Band Pair | Frequency Range | Description |
|-----------|-----------------|-------------|
| FB1-FB2 | 1000-2000 Hz | Low frequency range |
| FB2-FB3 | 1414-2828 Hz | Low-mid frequency range |
| FB3-FB4 | 2000-4000 Hz | Mid frequency range |
| FB4-FB5 | 2828-5657 Hz | Mid-high frequency range |
| FB5-FB6 | 4000-8000 Hz | High frequency range |
| FB6-FB7 | 5657-11314 Hz | Very high frequency range |
| FB7-FB8 | 8000-16000 Hz | Ultra-high frequency range |

#### Modulation Types

- **`noise`**: Replaces target frequency bands with spectrally matched noise
- **`amp`**: Applies amplitude modulation to target frequency bands
- **`phase`**: Applies phase/frequency modulation to target frequency bands

#### Hearing Loss Profiles

- **`NH`**: Normal Hearing (no correction)
- **`MildHL`**: Mild Hearing Loss (15 dB max correction)
- **`ModHL`**: Moderate Hearing Loss (30 dB max correction)
- **`SevHL`**: Severe Hearing Loss (45 dB max correction)

#### Example Commands

```bash
# Basic: Generate amplitude modulation for 3 kHz tinnitus
python tinnitus_sound_therapy.py -f 3000 -m amp

# Multiple modulations for 2 kHz tinnitus with mild hearing loss
python tinnitus_sound_therapy.py -f 2000 -m amp phase --hearing-profile NH MildHL

# 4 kHz tinnitus, all hearing profiles, save to specific directory
python tinnitus_sound_therapy.py -f 4000 --hearing-profile NH MildHL ModHL SevHL -o therapy_stimuli

# Generate multiple files per category for testing
python tinnitus_sound_therapy.py -f 1500 -n 3 --prefix test_

# Short duration for testing (16 seconds instead of 1 hour)
python tinnitus_sound_therapy.py -f 6000 -d 16 -o test_output

# Generate all experimental stimuli (warning: creates 84 large files!)
python tinnitus_sound_therapy.py --all -o full_experiment

# Use random filenames for blinded studies
python tinnitus_sound_therapy.py -f 8000 --random-names
```

#### Output Files

Generated files follow the naming pattern: `{modulation}_{band}_{hearing_profile}_{number}.wav`

Examples:
- `amp_FB3_NH_1.wav` - Amplitude modulation, FB3-FB4 bands, Normal Hearing
- `noise_FB5_MildHL_2.wav` - Noise modulation, FB5-FB6 bands, Mild Hearing Loss
- `phase_FB7_SevHL_1.wav` - Phase modulation, FB7-FB8 bands, Severe Hearing Loss

### 2. Hearing Assessment Stimuli (`hearing_assessment_stimuli.py`)

This tool generates stimuli for hearing assessment and tinnitus frequency estimation.

#### Basic Usage

```bash
# Generate all 8 hearing assessment files
python hearing_assessment_stimuli.py

# Using uv
uv run python hearing_assessment_stimuli.py
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Output directory | Current directory |
| `-f, --format` | Audio format: `wav`, `flac`, `ogg`, `mp4`* | wav |
| `-q, --quiet` | Suppress progress output | False |

*Note: MP4 format is automatically converted to WAV due to library limitations.

#### Example Commands

```bash
# Generate in current directory
python hearing_assessment_stimuli.py

# Generate in specific directory
python hearing_assessment_stimuli.py -o hearing_tests

# Generate FLAC files quietly
python hearing_assessment_stimuli.py -f flac --quiet

# Generate with progress output
python hearing_assessment_stimuli.py -o assessment_stimuli
```

#### Output Files

The tool generates 8 files (2 stimulus types × 4 hearing profiles):

**Pure Tone (PT) Files:**
- `Hearing_Tinnitus_Estimation_Stimuli_PT_NH.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_PT_MildHL.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_PT_ModHL.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_PT_SevHL.wav`

**Narrowband Noise (NBN) Files:**
- `Hearing_Tinnitus_Estimation_Stimuli_NBN_NH.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_NBN_MildHL.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_NBN_ModHL.wav`
- `Hearing_Tinnitus_Estimation_Stimuli_NBN_SevHL.wav`

Each file contains 17 test frequencies from 1-16 kHz in 1/4 octave steps, with 1-second stimuli separated by 1-second silent intervals.

### Testing and Validation

Run the test suite to verify correct installation and functionality:

```bash
# With pip
python test_implementation.py

# With uv
uv run python test_implementation.py
```

## Citation

If you use this code in research, please cite the original work:

```
Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
DOI: https://doi.org/10.25405/data.ncl.27109693
License: CC BY-NC-SA 4.0
```

## License

This Python port follows the same license as the original work: CC BY-NC-SA 4.0

## Support

For questions about the Python implementation, create an issue in this repository.
For questions about the research methodology, contact the original authors.