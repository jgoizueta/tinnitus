# Tinnitus Sound Therapy - Python Port

This is a Python port of the MatLab implementation for generating sound files targeted to patients with tonal tinnitus of specific frequencies, based on the research by Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., and Sedley, W.

## Original Research

- **Authors**: Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
- **Contact**: Yukhnovich, E.A. ney14@newcastle.ac.uk
- **DOI**: https://doi.org/10.25405/data.ncl.27109693
- **License**: CC BY-NC-SA 4.0
- **Last updated**: 15/10/2024

## Overview

This Python implementation provides the same functionality as the original MatLab code for creating modulated sounds used in tinnitus therapy experiments. The therapy uses spectral ripple sound modulation with three conditions:

1. **Amplitude modulation**: Modulates the amplitude of near-tinnitus frequencies
2. **Frequency/Phase modulation**: Modulates the phase of near-tinnitus frequencies
3. **Notched sound modulation**: Replaces near-tinnitus frequencies with spectrally matched noise

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.21.0
- soundfile>=0.10.0
- scipy>=1.7.0

### Files

- `tinnitus_sound_therapy.py` - Main Python module with all functions
- `test_implementation.py` - Test suite to verify functionality
- `requirements.txt` - Python dependencies

## Usage

### Quick Start

```python
from tinnitus_sound_therapy import create_example_stimulus

# Create a simple amplitude-modulated stimulus
stimulus = create_example_stimulus(
    f0=150,                    # Fundamental frequency (Hz)
    mod_type='amp',           # Modulation type ('amp', 'phase', 'noise', 'none')
    duration=4.0,             # Duration in seconds
    output_filename='example.wav'
)
```

### Core Functions

#### `mod_ripple()` - Core modulation function

```python
from tinnitus_sound_therapy import mod_ripple

# Set up frequency bands
fbands = [[1000, 1414], [1414, 2000], [2000, 2828], [2828, 4000]]
fbanddb = [0, 0, 0, 0]  # No hearing loss correction
fband_mod = [1, 2]      # Modulate bands 1 and 2

stimulus = mod_ripple(
    f0=150,                 # Fundamental frequency (Hz)
    rand_phase=0,           # 0=sine phase, 1=random phase
    dur=4.0,                # Duration (s)
    loud=0.1,               # Loudness scaling
    ramp=1.0,               # Ramp duration (s)
    tmr=1.0,                # Temporal modulation rate (Hz)
    smr=[1.5, 7.5],         # Spectral modulation rate range (cyc/oct)
    scyc=8,                 # SMR cycle duration (s)
    fbands=fbands,          # Frequency bands
    fbanddb=fbanddb,        # Band amplitude adjustments (dB)
    fband_mod=fband_mod,    # Bands to modulate (0-indexed)
    mod_type='amp',         # Modulation type
    normfreq=0              # 0=constant amplitude, 1=1/f scaling
)
```

#### `generate_full_experiment_stimuli()` - Generate complete stimulus set

```python
from tinnitus_sound_therapy import generate_full_experiment_stimuli

# Generate full experimental stimuli (WARNING: creates many files)
generate_full_experiment_stimuli(
    loud=0.1,               # Loudness scaling
    files_per_category=1,   # Number of files per category
    randnames=True,         # Use random filenames
    name_key_file='Random_File_Names_Key'  # Key file for randomization
)
```

#### `generate_hearing_tinnitus_estimation_stimuli()` - Generate estimation stimuli

```python
from tinnitus_sound_therapy import generate_hearing_tinnitus_estimation_stimuli

# Generate pure tone and narrowband noise stimuli for hearing assessment
generate_hearing_tinnitus_estimation_stimuli()
```

### Utility Functions

#### `farpn()` - Fixed-amplitude random-phase noise

```python
from tinnitus_sound_therapy import farpn

noise = farpn(
    srate=44100,    # Sample rate (Hz)
    low_f=1000,     # Lower frequency limit (Hz)
    high_f=2000,    # Upper frequency limit (Hz)
    dur=1.0         # Duration (s)
)
```

#### `wind_ramp()` - Apply onset/offset ramps

```python
from tinnitus_sound_therapy import wind_ramp
import numpy as np

signal = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/44100))
ramped_signal = wind_ramp(
    srate=44100,    # Sample rate (Hz)
    wdms=10,        # Ramp duration (ms)
    x=signal        # Input signal
)
```

## Function Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `mod_ripple()` | Create harmonic complexes with band-limited modulation |
| `generate_full_experiment_stimuli()` | Generate complete experimental stimulus set |
| `generate_hearing_tinnitus_estimation_stimuli()` | Generate hearing estimation stimuli |
| `create_example_stimulus()` | Create simple example stimulus |

### Utility Functions

| Function | Description |
|----------|-------------|
| `farpn()` | Fixed-amplitude random-phase noise |
| `wind_ramp()` | Apply onset/offset ramps |
| `phase_randomise()` | Phase randomization for RIN stimuli |
| `freqfilter()` | Frequency domain filtering |

## Modulation Types

- **'amp'**: Amplitude modulation - varies the amplitude of target frequency bands
- **'phase'**: Phase modulation - varies the phase of target frequency bands
- **'noise'**: Noise replacement - replaces target bands with spectrally matched noise
- **'none'**: No modulation - plain harmonic complex

## Testing

Run the test suite to verify the implementation:

```bash
python test_implementation.py
```

This will test:
- Basic function functionality
- Parameter validation
- Audio generation
- File output

## Example Output

Running the main script generates example stimuli:

```bash
python tinnitus_sound_therapy.py
```

This creates:
- `amplitude_modulation_example.wav`
- `phase_modulation_example.wav`
- `noise_modulation_example.wav`

## Differences from MatLab Version

The Python port maintains functional equivalence with the original MatLab code while using Python-native libraries:

- **NumPy** for numerical computations (replaces MatLab arrays)
- **SoundFile** for audio I/O (replaces MatLab audiowrite)
- **SciPy** for signal processing functions
- 0-based indexing (Python) vs 1-based indexing (MatLab)
- Function signatures use Python type hints
- Error handling uses Python exceptions

## Research Application

This implementation supports research into tinnitus therapy using:

1. **Amplitude modulation**: Modulating amplitude near tinnitus frequency
2. **Frequency modulation**: Modulating phase/frequency near tinnitus frequency
3. **Notched sounds**: Replacing tinnitus frequency bands with noise

The therapy protocol involves:
- 6 weeks listening to active sound (modulation near tinnitus frequency)
- 3 week washout period
- 6 weeks listening to sham sound (modulation far from tinnitus frequency)
- 3 week washout period

## Citation

If you use this code in research, please cite the original work:

```
Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
DOI: https://doi.org/10.25405/data.ncl.27109693
License: CC BY-NC-SA 4.0
```

* Original Paper: https://www.sciencedirect.com/science/article/pii/S0378595525001534
* Original MatLab programs: https://data.ncl.ac.uk/articles/software/Tinnitus_Spectral_Ripple_Sound_Therapy_Files/27109693

## License

This Python port follows the same license as the original work: CC BY-NC-SA 4.0

## Support

For questions about the Python implementation, create an issue in this repository.
For questions about the research methodology, contact the original authors.