# Tinnitus Sound Therapy


This is a Python port of the MatLab code for generating audio stimulus files for tinnitus sound therapy based on the [research by Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., and Sedley, W.](https://www.sciencedirect.com/science/article/pii/S0378595525001534)

## Original Research

- **Authors**: Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
- **DOI**: https://doi.org/10.25405/data.ncl.27109693
- **License**: CC BY-NC-SA 4.0
- **Last updated**: 15/10/2024

## Overview

The Python program `tinnitus_sound_therapy.py` provides a comprehensive CLI tool for generating audio files for tinnitus research and therapy. It can generate:

* **Therapeutic stimuli**: Based on the original `Generate_Full_Experiment_Stimuli` MatLab implementation, generating the full set of experimental files or specific files for given tinnitus frequency, hearing loss, etc.
* **Hearing assessment stimuli**: Based on the original `Hearing_Tin_Stim_Generation.m` MatLab file, generating files for assessment of tinnitus frequency and hearing loss slope.


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

- `tinnitus_sound_therapy.py` - Main CLI tool for generating therapeutic and assessment stimuli
- `test_implementation.py` - Test suite to verify functionality
- `requirements.txt` - Python dependencies for pip
- `pyproject.toml` - Modern Python project configuration for uv

## Usage

This project provides a comprehensive CLI tool for generating different types of audio stimuli for tinnitus research and therapy.

### 1. Therapeutic Stimuli Generation

The main tool can generate therapeutic audio stimuli with various modulation types using frequency or band-specific targeting.

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
| `--assessment-stimuli` | Generate hearing assessment stimuli (8 files for frequency estimation) | - |
| `-f, --frequency` | Tinnitus frequency in Hz (determines frequency band) | - |
| `--mod-band` | Direct frequency band selection (FB1-FB7, alternative to --frequency) | - |
| `-m, --modulation` | Modulation type(s): `noise`, `amp`, `phase` | All types |
| `--hearing-profile` | Hearing loss profile(s): `NH`, `MildHL`, `ModHL`, `SevHL` | NH only |
| `-n, --files-per-category` | Number of files per category | 1 |
| `-o, --output-dir` | Output directory | Current directory |
| `--format` | Audio file format for assessment stimuli: `wav`, `flac`, `ogg`, `mp4` | wav |
| `--prefix` | Prefix for generated filenames | None |
| `--random-names` | Use random filenames instead of descriptive names | False |
| `-q, --quiet` | Suppress progress output for assessment stimuli | False |
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

#### Frequency Band Selection

**Automatic Selection (`-f, --frequency`)**
The tool automatically determines the optimal frequency band pair based on the tinnitus frequency using logarithmic centering. For example:
```bash
python tinnitus_sound_therapy.py -f 3000  # Automatically selects FB4-FB5
```

**Manual Selection (`--mod-band`)**
For clinical applications, you may need to manually select a specific frequency band, e.g. when the automatically selected _best_ band for the tinnitus frequency contains frequencies not heard by the subject.

It can also be useful for generating _sham_ files for clinical studies (where the modulated frequencies are not close to the tinnitus frequency).

```bash
# Direct band selection when automatic choice isn't suitable
python tinnitus_sound_therapy.py --mod-band FB3  # Force FB3-FB4 bands

# Example: 4 kHz tinnitus might auto-select FB4-FB5, but patient has hearing loss there
python tinnitus_sound_therapy.py -f 4000        # Auto-selects FB4-FB5 (2828-5657 Hz)
python tinnitus_sound_therapy.py --mod-band FB3 # Manual override to FB3-FB4 (2000-4000 Hz)
```

**Note**: Only one frequency selection method can be used per command (`--frequency` OR `--mod-band`, not both).

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

# Direct band selection examples
python tinnitus_sound_therapy.py --mod-band FB3 -m amp               # Direct FB3-FB4 selection
python tinnitus_sound_therapy.py --mod-band FB5 --hearing-profile SevHL -d 30  # FB5-FB6 with severe HL correction

# Clinical scenario: Patient has 4 kHz tinnitus but high-frequency hearing loss
python tinnitus_sound_therapy.py -f 4000 -m amp                     # Auto-selects FB4-FB5 (2828-5657 Hz)
python tinnitus_sound_therapy.py --mod-band FB3 -m amp              # Manual override to FB3-FB4 (2000-4000 Hz)

# Generate hearing assessment stimuli for frequency estimation
python tinnitus_sound_therapy.py --assessment-stimuli               # Generate all 8 assessment files
```

#### Output Files

Generated files follow the naming pattern: `{modulation}_{band}_{hearing_profile}_{number}.wav`

Examples:
- `amp_FB3_NH_1.wav` - Amplitude modulation, FB3-FB4 bands, Normal Hearing
- `noise_FB5_MildHL_2.wav` - Noise modulation, FB5-FB6 bands, Mild Hearing Loss
- `phase_FB7_SevHL_1.wav` - Phase modulation, FB7-FB8 bands, Severe Hearing Loss

### 2. Hearing Assessment Stimuli (`--assessment-stimuli`)

The main CLI tool can also generate stimuli for hearing assessment and tinnitus frequency estimation using the `--assessment-stimuli` option.

#### Basic Usage

```bash
# Generate all 8 hearing assessment files
python tinnitus_sound_therapy.py --assessment-stimuli

# Using uv
uv run python tinnitus_sound_therapy.py --assessment-stimuli
```

#### Additional Options for Assessment Stimuli

When using `--assessment-stimuli`, these additional options are available:

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir` | Output directory | Current directory |
| `--format` | Audio format: `wav`, `flac`, `ogg`, `mp4`* | wav |
| `-q, --quiet` | Suppress progress output | False |

*Note: MP4 format is automatically converted to WAV due to library limitations.

#### Example Commands

```bash
# Generate in current directory
python tinnitus_sound_therapy.py --assessment-stimuli

# Generate in specific directory
python tinnitus_sound_therapy.py --assessment-stimuli -o hearing_tests

# Generate FLAC files quietly
python tinnitus_sound_therapy.py --assessment-stimuli --format flac -q

# Generate with progress output in specific directory
python tinnitus_sound_therapy.py --assessment-stimuli -o assessment_stimuli
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