"""
Tinnitus Sound Therapy - Python port of MatLab implementation

This module provides functionality to generate sound files for tinnitus therapy
based on spectral ripple sound therapy methods.

Ported from MatLab code by Yukhnovich, E.A., Harrison, S., Wray, N., Alter, K., Sedley, W.
Original article: DOI: https://doi.org/10.25405/data.ncl.27109693
"""

import numpy as np
import soundfile as sf
from typing import List, Tuple, Union, Optional
import random


def farpn(srate: int, low_f: float, high_f: float, dur: float) -> np.ndarray:
    """
    Fixed-Amplitude Random-Phase Noise

    Produces fixed-amplitude random-phase noise with specified passband

    Args:
        srate: sampling rate (Hz)
        low_f: lower limit of passband (Hz)
        high_f: upper limit of passband (Hz)
        dur: duration of output stimulus (s)

    Returns:
        output stimulus (1D array)

    Originally coded in MatLab by Tim Griffiths, 2004
    Part of Pitch Stimulus Design Toolbox (https://github.com/lerud/lerud.github.io/)
    """
    # Calculate number of samples needed
    npts_needed = int(dur * srate)

    # Make sure npts is even for proper FFT structure
    npts = int(np.floor(npts_needed / 2) * 2)
    fbin = srate / npts

    mag = np.ones(npts // 2)
    mag[:int(np.floor(low_f / fbin))] = 0
    mag[int(np.ceil(high_f / fbin)):] = 0

    phase = 2 * np.pi * np.random.rand(npts // 2 - 1)
    allphase = np.concatenate([[0], phase, [0], -np.flip(phase)])
    allmag = np.concatenate([mag, np.flip(mag)])

    rect = allmag * np.exp(1j * allphase)
    sig = np.real(np.fft.ifft(rect))

    # Trim to exact duration if needed
    if len(sig) > npts_needed:
        sig = sig[:npts_needed]

    # Set RMS value to 0.1
    sig_std = np.std(sig)
    if sig_std > 0:
        x = sig / (10 * sig_std)
    else:
        x = sig  # If std is 0, signal is already at zero

    return x


def wind_ramp(srate: int, wdms: float, x: np.ndarray) -> np.ndarray:
    """
    Windowing function

    Create onset and offset ramps at either end of a stimulus to remove
    clicks at stimulus onset or transition.

    Args:
        srate: stimulus sampling rate (Hz)
        wdms: length of onset and offset ramps (ms)
        x: input stimulus (1D array)

    Returns:
        output stimulus (1D array)

    Part of Pitch Stimulus Design Toolbox
    Tim Griffiths, Newcastle Auditory Group, UK
    December 2010
    """
    npts = len(x)
    wds = int(np.round(2 * wdms / 1000 * srate))
    if wds % 2 != 0:
        wds = wds + 1

    # Ensure ramp doesn't exceed signal length
    wds = min(wds, npts)
    if wds == 0:
        return x.copy()

    w = np.linspace(-np.pi/2, 1.5*np.pi, wds)
    w = (np.sin(w) + 1) / 2

    x_out = x.copy()
    ramp_len = int(np.round(wds/2))
    ramp_len = min(ramp_len, npts//2)  # Don't let ramps overlap

    if ramp_len > 0:
        x_out[:ramp_len] = x_out[:ramp_len] * w[:ramp_len]

        if srate == 48828:
            end_start = max(0, min(npts-ramp_len, wds//2-1))
            x_out[npts-ramp_len:npts] = x_out[npts-ramp_len:npts] * w[end_start:end_start+ramp_len]
        else:
            end_start = max(0, min(wds//2, wds-ramp_len))
            x_out[npts-ramp_len:npts] = x_out[npts-ramp_len:npts] * w[end_start:end_start+ramp_len]

    return x_out


def phase_randomise(stim_in: np.ndarray, srate: int, time_bin: float) -> np.ndarray:
    """
    Phase randomisation function

    Designed for use with Regular Interval Noise (RIN) stimuli

    Converts input stimulus into frequency domain, randomises phase
    values while maintaining power spectrum over time, then returns
    stimulus to time domain.

    Args:
        stim_in: input waveform (1D array)
        srate: stimulus sampling rate (Hz)
        time_bin: time bin length (ms) used for each phase randomisation

    Returns:
        output waveform (1D array)

    Part of Pitch Stimulus Design Toolbox
    Will Sedley, Newcastle Auditory Group, UK
    December 2010
    """
    time_bin = int(2 * np.round(time_bin * srate / 2000))
    stim_in = stim_in[:len(stim_in) - (len(stim_in) % time_bin)]
    stim_out_a = np.zeros_like(stim_in)
    stim_out_b = np.zeros_like(stim_in)
    nbins = len(stim_in) // time_bin

    # Create Hanning window
    hanwin = np.hanning(time_bin)
    hanstart = np.concatenate([np.ones(time_bin//2), hanwin[time_bin//2:]])
    hanend = np.flip(hanstart)

    for cur_bin in range(int(np.floor(nbins))):
        bin_start = cur_bin * time_bin
        bin_end = (cur_bin + 1) * time_bin
        bin_f = np.fft.fft(stim_in[bin_start:bin_end])
        bin_f = np.abs(bin_f)
        bin_phase = 2 * np.pi * np.random.rand(time_bin//2 - 1)
        bin_phase = np.concatenate([[0], bin_phase, [0], -np.flip(bin_phase)])

        if cur_bin == 0:
            stim_out_a[bin_start:bin_end] = hanstart * np.real(np.fft.ifft(bin_f * np.exp(1j * bin_phase)))
        elif cur_bin == int(np.floor(nbins)) - 1:
            stim_out_a[bin_start:bin_end] = hanend * np.real(np.fft.ifft(bin_f * np.exp(1j * bin_phase)))
        else:
            stim_out_a[bin_start:bin_end] = hanwin * np.real(np.fft.ifft(bin_f * np.exp(1j * bin_phase)))

    for cur_bin in range(int(np.floor(nbins - 1))):
        bin_start = int((cur_bin + 0.5) * time_bin)
        bin_end = int((cur_bin + 1.5) * time_bin)
        bin_f = np.fft.fft(stim_in[bin_start:bin_end])
        bin_f = np.abs(bin_f)
        bin_phase = 2 * np.pi * np.random.rand(time_bin//2 - 1)
        bin_phase = np.concatenate([[0], bin_phase, [0], -np.flip(bin_phase)])
        stim_out_b[bin_start:bin_end] = hanwin * np.real(np.fft.ifft(bin_f * np.exp(1j * bin_phase)))

    stim_out = stim_out_a + stim_out_b
    return stim_out


def freqfilter(srate: int, low_f: float, high_f: float, stim_in: np.ndarray) -> np.ndarray:
    """
    Frequency domain filtering

    Transforms input stimulus into frequency domain, performs filtering
    and returns to time domain with zero phase delay.

    Args:
        srate: sampling rate (Hz)
        low_f: lower limit of band pass filter (Hz)
        high_f: upper limit of band pass filter (Hz)
        stim_in: input stimulus (1D array)

    Returns:
        output stimulus (1D array)

    Originally by Tim Griffiths, 2002
    Part of Pitch Stimulus Design Toolbox
    """
    s1 = stim_in.copy()
    if len(s1) % 2 != 0:
        s1 = s1[:-1]

    npts = len(s1)
    fbin = srate / npts
    f = np.fft.fft(s1)

    f[:int(np.fix(low_f / fbin))] = 0
    f[int(np.fix(high_f / fbin) + 1):npts - int(np.fix(high_f / fbin))] = 0
    f[npts - int(np.fix(low_f / fbin)):npts] = 0

    s2 = np.real(np.fft.ifft(f))
    return s2


def mod_ripple(f0: float, rand_phase: int, dur: float, loud: float, ramp: float,
               tmr: float, smr: List[float], scyc: float, fbands: List[List[float]],
               fbanddb: List[float], fband_mod: List[int], mod_type: str,
               normfreq: int = 0) -> np.ndarray:
    """
    Function to create harmonic complexes with band-limited amplitude or phase modulation
    based on variable spectral modulation rate dynamic spectral ripples

    Args:
        f0: fundamental frequency of harmonic complex carrier (Hz) - suggest 96 to 256
        rand_phase: If 0, carrier harmonics are in sine phase; if nonzero then harmonics are in random phase
        dur: stimulus duration (s) - suggest 4
        loud: loudness (i.e. amplitude) scaling factor - suggest 0.1
        ramp: duration of each onset/offset ramp (s) - suggest 1
        tmr: temporal modulation rate of superimposed ripple (Hz) - suggest 1
        smr: [minimum, maximum] spectral modulation rate (cyc/oct) - suggest [1.5,7.5]
        scyc: cycle duration for changing SMR (higher value = slower rate of change) (s) - suggest 8
        fbands: lower and upper limits (Hz) of specified frequency bands - [[lo1, hi1], [lo2, hi2], ...]
        fbanddb: amplitude scaling value (dB) for each specified frequency band
        fband_mod: indices of frequency bands to receive modulation (0-indexed)
        mod_type: type of modulation: 'phase', 'amp', 'noise', or 'none'
        normfreq: If 1, scale amplitude of each harmonic to 1/f distribution; If 0 use constant amplitude

    Returns:
        final stimulus (harmonic complex containing specified modulation)

    William Sedley - Last updated September 2024
    """
    # Non-modifiable variables
    depth = 1  # Modulation depth (0-1 range)
    srate = 44100  # Sample rate (Hz)
    phase_off = np.random.rand() * 2 * np.pi  # Starting offset for modulation waveform (rad)
    phase_s = np.random.rand() * 2 * np.pi  # Starting offset for ripple spacing parameter (rad)
    t = np.arange(1/srate, dur + 1/srate, 1/srate)  # Time vector for stimulus

    # Determine harmonic details
    fn = [] # Number of the harmonics of f0 (frequency of harmonic = fn * f0)
    fmod = [] # For each harmonic, whether it is to be modulated (1) or not (0)
    fintens = [] # Relative intensity of harmonics
    band_ind = [[] for _ in range(len(fbands))] # Indices of harmonics (0-based position in fn) belonging to each frequency band
    ind_tot = 0 # number of harmonics (elements in fn)
    nf = len(fbands)

    for f_index in range(nf):
        frange_tmp = fbands[f_index]
        fntmp = list(range(int(np.ceil(frange_tmp[0]/f0)), int(np.floor((frange_tmp[1]-1)/f0)) + 1))
        fn.extend(fntmp)  # Numbers of the harmonics within the current band frange_tmp

        if f_index in fband_mod:
            fmod.extend([1] * len(fntmp))  # Yes or no to harmonic modulation
        else:
            fmod.extend([0] * len(fntmp))

        fintens.extend([10**(fbanddb[f_index]/20)] * len(fntmp))  # Relative intensity of harmonics
        nind_tmp = len(fntmp)
        band_ind[f_index] = list(range(ind_tot, ind_tot + nind_tmp))  # Indices of harmonics belonging to each frequency band
        ind_tot = ind_tot + nind_tmp

    fn = np.array(fn)
    fmod = np.array(fmod)
    fintens = np.array(fintens)
    fi = fn * f0  # Frequency of harmonics

    if normfreq:  # 1/f intensity scaling of harmonics
        fintens = fintens / fi

    # Determine modulation parameters
    s = np.mean(smr) + (np.abs(np.diff(smr))/2) * np.sin(phase_s + 2*np.pi*(1/scyc)*t)  # Variable spectral modulation rate waveform
    fmodind = np.where(fmod)[0]  # Indices of harmonics to be modulated
    fimod = fi[fmodind]  # Frequencies of harmonics to be modulated
    f_log = np.log2(fimod/np.min(fimod))
    f_log = f_log - np.mean(f_log)  # Log-scaled, mean-centred, modulated harmonics

    tmat = np.tile(t, (len(f_log), 1))
    fmat = np.tile(f_log.reshape(-1, 1), (1, len(t)))
    tmat_full = np.tile(t, (len(fi), 1))
    smat = np.tile(s, (len(f_log), 1))
    intmat = np.tile(fintens.reshape(-1, 1), (1, len(t)))

    # Generate ripples and stimuli
    if rand_phase:
        pmat = np.tile((np.random.rand(len(fi)) * 2 * np.pi).reshape(-1, 1), (1, len(t)))  # Phase offset for each harmonic
    else:
        pmat = np.tile((np.zeros(len(fi)) * 2 * np.pi).reshape(-1, 1), (1, len(t)))

    fimat = np.tile(fi.reshape(-1, 1), (1, len(t)))

    if mod_type == 'phase':  # Phase shift modulation
        mmat = np.zeros_like(fimat)
        mmat[fmodind, :] = np.pi * (1 + depth * np.sin(2*np.pi*(phase_off + tmr*tmat + smat*fmat))) / 2  # Modulate only specified band
        stim_mat = np.sin(2 * np.pi * fimat * tmat_full + pmat + mmat) * intmat  # Stimulus waveforms for individual harmonics
    elif mod_type == 'amp':  # Amplitude modulation
        mmat = np.ones_like(fimat)
        mmat[fmodind, :] = (1 + depth * np.sin(2*np.pi*(phase_off + tmr*tmat + smat*fmat)))  # Modulate only specified band
        stim_mat = mmat * np.sin(2 * np.pi * fimat * tmat_full + pmat) * intmat  # Stimulus waveforms for individual harmonics
    elif mod_type == 'noise':  # Bandpass noise
        stim_mat = np.sin(2 * np.pi * fimat * tmat_full + pmat) * intmat  # Stimulus waveforms for individual harmonics
        stim_mat_tmp = stim_mat[fmodind, :]
        stim_tmp = np.sum(stim_mat_tmp, axis=0)
        stmp_std = np.std(stim_tmp)
        noise_tmp = farpn(srate, np.min(fimod), np.max(fimod), dur)  # Noise to fill in gap
        noise_std = np.std(noise_tmp)
        if noise_std > 0 and stmp_std > 0:
            noise_tmp = noise_tmp * stmp_std / noise_std  # Scale to same RMS as harmonic band being replaced
        elif stmp_std == 0:
            noise_tmp = np.zeros_like(noise_tmp)  # If target is silent, make noise silent too
        stim_mat[fmodind, :] = 0  # Silence harmonics from noise band
        stim_mat[np.min(fmodind), :] = noise_tmp  # Add noise to noise band
        mmat = np.ones_like(fimat)
        mmat[fmodind, :] = 0
    elif mod_type == 'none':  # Plain harmonic complex
        stim_mat = np.sin(2 * np.pi * fimat * tmat_full + pmat) * intmat  # Stimulus waveforms for individual harmonics
        mmat = np.ones_like(fimat)
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")

    # Calculate RMS of whole stimulus to scale band-specific stimuli
    stim_loud = np.sum(stim_mat, axis=0)
    stim_scale = loud / (10 * np.std(stim_loud))

    # Produce stimuli within specific frequency bands
    stim_band = np.zeros((nf, len(t)))
    for f in range(nf):
        if band_ind[f]:  # Only process if band has harmonics
            stim_band[f, :] = np.sum(stim_mat[band_ind[f], :], axis=0)
            stim_band[f, :] = wind_ramp(srate, ramp * 1000, stim_band[f, :])
    stim_band = stim_band * stim_scale

    # Flatten across bands to produce final stimulus
    stim_out = np.sum(stim_band, axis=0)

    return stim_out


def generate_single_experiment_stimulus(mod_type: str, fband_index: int, hearing_profile: str,
                                      file_number: int = 1, loud: float = 0.1,
                                      output_dir: str = ".", randnames: bool = False,
                                      name_key_file: str = 'Random_File_Names_Key',
                                      duration_seconds: float = 3600) -> str:
    """
    Generate a single experimental stimulus file for specific parameters

    Args:
        mod_type: modulation type ('noise', 'amp', 'phase')
        fband_index: frequency band index (0-6 for FB1-FB7)
        hearing_profile: hearing correction profile ('NH', 'MildHL', 'ModHL', 'SevHL')
        file_number: file number for this category (default: 1)
        loud: loudness scaling factor (default: 0.1)
        output_dir: output directory (default: current directory)
        randnames: whether to use random file names (default: False)
        name_key_file: filename for the randomization key
        duration_seconds: total duration of generated file in seconds (default: 3600 = 1 hour)

    Returns:
        filepath of generated file

    William Sedley - Last updated September 2024
    """
    import os

    # Validate inputs
    valid_mod_types = ['noise', 'amp', 'phase']
    valid_hearing_profiles = ['NH', 'MildHL', 'ModHL', 'SevHL']

    if mod_type not in valid_mod_types:
        raise ValueError(f"mod_type must be one of {valid_mod_types}, got '{mod_type}'")
    if fband_index < 0 or fband_index > 6:
        raise ValueError(f"fband_index must be 0-6 (FB1-FB7), got {fband_index}")
    if hearing_profile not in valid_hearing_profiles:
        raise ValueError(f"hearing_profile must be one of {valid_hearing_profiles}, got '{hearing_profile}'")

    # Parameters from MatLab implementation
    srate = 44100
    dur_range = [4, 4]
    ramp_prop = 0.25
    f0_range = [96, 256]

    # Calculate number of stimuli needed for target duration
    # Each stimulus is ~4 seconds, so n_per_file = duration_seconds / 4
    n_per_file = max(1, int(duration_seconds / 4))  # At least 1 stimulus

    # Frequency bands setup - exactly matching MatLab
    fb = band_separation_frequencies() 
    fbands = frequency_bands(fb)

    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 2.25, 0.25), [2, 2, 2, 2]]) / 2
    hl_corr_temp_fbands = hl_corr_temp[1::2]  # Every second element starting from index 1

    hl_corr_vals = {'NH': 0, 'MildHL': 15, 'ModHL': 30, 'SevHL': 45}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    fname_tmp = f"{mod_type}_FB{fband_index+1}_{hearing_profile}_{file_number}.wav"

    if randnames:
        # For random names, we'd need to manage the global counter
        # For simplicity, use timestamp-based random name for single file generation
        import time
        random_id = int(time.time() * 1000) % 10000
        rname_tmp = f"Tin_Mod_File_{random_id}.wav"
        filepath = os.path.join(output_dir, rname_tmp)
        # Could save the mapping to file_key if needed
    else:
        filepath = os.path.join(output_dir, fname_tmp)

    print(f"Generating {mod_type} modulation, FB{fband_index+1}, {hearing_profile} profile...")

    # Generate stimulus sequence
    stim = []
    for s in range(n_per_file):
        dur_tmp = np.round(10 * (min(dur_range) + np.random.rand() * np.diff(dur_range)[0])) / 10
        f0_tmp = int(np.round(min(f0_range) + np.random.rand() * np.diff(f0_range)[0]))
        fband_db = hl_corr_temp_fbands * hl_corr_vals[hearing_profile]
        fband_mod = [fband_index, fband_index + 1]  # Use adjacent frequency bands

        stim_tmp = mod_ripple(
            f0=f0_tmp,
            rand_phase=0,
            dur=dur_tmp,
            loud=loud,
            ramp=dur_tmp * ramp_prop,
            tmr=1,
            smr=[1.5, 7.5],  # Updated SMR values
            scyc=8,
            fbands=fbands,
            fbanddb=fband_db.tolist(),
            fband_mod=fband_mod,
            mod_type=mod_type,
            normfreq=0
        )
        stim.extend(stim_tmp)

    stim = np.array(stim)
    maxabs = np.max(np.abs(stim))
    if maxabs > 1:
        stim = stim / maxabs
        print('  Downscaling to prevent clipping.')

    # Save file
    sf.write(filepath, stim, srate)

    duration_min = len(stim) / srate / 60
    print(f"  Generated: {os.path.basename(filepath)} ({duration_min:.1f} minutes)")

    return filepath


def generate_full_experiment_stimuli_updated(loud: float = 0.1, files_per_category: int = 1,
                                           randnames: bool = False, name_key_file: str = 'Random_File_Names_Key') -> None:
    """
    Generate full set of experimental stimuli used in online tinnitus modulation study

    Updated version that matches the latest MatLab implementation (September 2024)
    This script exactly recreates stimuli used in original published trial of online therapy

    Args:
        loud: loudness scaling factor (default: 0.1) - needs optimizing
        files_per_category: number of files per category (default: 1) - higher number reduces repetitiveness
        randnames: whether to use random file names (default: False) - set to True for final rollout
        name_key_file: filename for the randomization key (default: 'Random_File_Names_Key')

    William Sedley - Last updated September 2024
    """
    srate = 44100
    mod_type = ['noise', 'amp', 'phase']
    dur_range = [4, 4]
    ramp_prop = 0.25
    f0_range = [96, 256]
    n_per_file = 900  # 60 minutes exactly

    # Frequency bands setup - exactly matching MatLab
    fb = band_separation_frequencies() 
    fbands = frequency_bands(fb)

    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 2.25, 0.25), [2, 2, 2, 2]]) / 2
    hl_corr_temp_fbands = hl_corr_temp[1::2]  # Every second element starting from index 1

    hl_corr_vals = [0, 15, 30, 45]  # Maximum (dB) correction difference between 8 kHz and 1 kHz
    hl_corr_labels = ['NH', 'MildHL', 'ModHL', 'SevHL']

    if randnames:
        ntot = len(mod_type) * (len(fbands) - 1) * len(hl_corr_vals) * files_per_category
        rnames = np.random.permutation(ntot) + 1  # +1 to match MatLab 1-indexing
        file_key = []

    print(f"Generating experimental stimuli...")
    print(f"Total files to generate: {len(mod_type) * (len(fbands) - 1) * len(hl_corr_vals) * files_per_category}")
    print(f"This will take some time...")

    file_count = 0
    total_files = len(mod_type) * (len(fbands) - 1) * len(hl_corr_vals) * files_per_category

    for m in range(len(mod_type)):
        for b in range(len(fbands) - 1):  # -1 because we use pairs of adjacent bands
            for h in range(len(hl_corr_vals)):
                for f in range(files_per_category):
                    # Use the single stimulus generation function
                    try:
                        filepath = generate_single_experiment_stimulus(
                            mod_type=mod_type[m],
                            fband_index=b,
                            hearing_profile=hl_corr_labels[h],
                            file_number=f + 1,
                            loud=loud,
                            output_dir=".",  # Current directory
                            randnames=randnames,
                            name_key_file=name_key_file
                        )

                        if randnames:
                            # For consistency with original behavior, track the mapping
                            import os
                            fname_tmp = f"{mod_type[m]}_FB{b+1}_{hl_corr_labels[h]}_{f+1}.wav"
                            file_key.append([fname_tmp, os.path.basename(filepath)])

                        file_count += 1
                        print(f"Progress: {file_count}/{total_files} files completed")

                    except Exception as e:
                        print(f"Error generating {mod_type[m]}_FB{b+1}_{hl_corr_labels[h]}_{f+1}: {e}")
                        continue

    if randnames:
        np.save(name_key_file, file_key)
        print(f"Randomization key saved to: {name_key_file}.npy")

    print("Full experimental stimuli generation completed!")


def test_experimental_stimuli_generation(n_per_file: int = 2) -> None:
    """
    Test version of experimental stimuli generation with minimal parameters

    Args:
        n_per_file: number of stimuli per file (default: 2 for testing)
    """
    print("Testing experimental stimuli generation (minimal version)...")

    srate = 44100
    mod_type = ['noise', 'amp']  # Test only 2 types
    dur_range = [0.5, 0.5]  # Shorter duration
    ramp_prop = 0.25
    f0_range = [150, 150]  # Fixed frequency
    loud = 0.1

    # Frequency bands setup - reduced set
    fb = band_separation_frequencies() 
    fbands = frequency_bands(fb)

    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 1.25, 0.25)]) / 2
    hl_corr_temp_fbands = hl_corr_temp[1::2][:len(fbands)]  # Match fbands length

    hl_corr_vals = [0, 15]  # Only 2 hearing loss conditions
    hl_corr_labels = ['NH', 'MildHL']

    total_files = len(mod_type) * (len(fbands) - 1) * len(hl_corr_vals)
    print(f"Will generate {total_files} test files...")

    file_count = 0

    for m in range(len(mod_type)):
        for b in range(len(fbands) - 1):
            for h in range(len(hl_corr_vals)):
                fname_tmp = f"test_{mod_type[m]}_FB{b+1}_{hl_corr_labels[h]}.wav"

                stim = []
                for s in range(n_per_file):
                    dur_tmp = np.round(10 * (min(dur_range) + np.random.rand() * np.diff(dur_range)[0])) / 10
                    f0_tmp = int(np.round(min(f0_range) + np.random.rand() * np.diff(f0_range)[0]))
                    fband_db = hl_corr_temp_fbands[:len(fbands)] * hl_corr_vals[h]
                    fband_mod = [b, b + 1]

                    stim_tmp = mod_ripple(
                        f0=f0_tmp,
                        rand_phase=0,
                        dur=dur_tmp,
                        loud=loud,
                        ramp=dur_tmp * ramp_prop,
                        tmr=1,
                        smr=[1.5, 7.5],
                        scyc=8,
                        fbands=fbands,
                        fbanddb=fband_db.tolist(),
                        fband_mod=fband_mod,
                        mod_type=mod_type[m],
                        normfreq=0
                    )
                    stim.extend(stim_tmp)

                stim = np.array(stim)
                maxabs = np.max(np.abs(stim))
                if maxabs > 1:
                    stim = stim / maxabs
                    print('Downscaling to prevent clipping.')

                sf.write(fname_tmp, stim, srate)
                file_count += 1
                print(f"Generated: {fname_tmp} ({file_count}/{total_files})")

    print(f"Test generation completed! Generated {file_count} files.")


def generate_full_experiment_stimuli(loud: float = 0.1, files_per_category: int = 1,
                                   randnames: bool = True, name_key_file: str = 'Random_File_Names_Key') -> None:
    """
    Generate full set of experimental stimuli used in online tinnitus modulation study

    This function exactly recreates stimuli used in original published trial of online therapy

    Args:
        loud: loudness scaling factor (default: 0.1)
        files_per_category: number of files per category (default: 1)
        randnames: whether to use random file names (default: True)
        name_key_file: filename for the randomization key (default: 'Random_File_Names_Key')

    William Sedley - Last updated September 2024
    """
    srate = 44100
    mod_type = ['noise', 'amp', 'phase']
    dur_range = [4, 4]
    ramp_prop = 0.25
    f0_range = [96, 256]
    n_per_file = 900  # 60 minutes exactly

    fb = band_separation_frequencies() 
    fbands = frequency_bands(fb)
    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 2.25, 0.25), [2, 2, 2, 2]]) / 2
    hl_corr_temp_fbands = hl_corr_temp[1::2]  # Every second element starting from index 1
    hl_corr_vals = [0, 15, 30, 45]  # Maximum (dB) correction difference between 8 kHz and 1 kHz
    hl_corr_labels = ['NH', 'MildHL', 'ModHL', 'SevHL']

    if randnames:
        ntot = len(mod_type) * (len(fbands) - 1) * len(hl_corr_vals) * files_per_category
        rnames = np.random.permutation(ntot) + 1  # +1 to match MatLab 1-indexing
        file_key = []

    for m in range(len(mod_type)):
        for b in range(len(fbands) - 1):
            for h in range(len(hl_corr_vals)):
                for f in range(files_per_category):
                    fname_tmp = f"{mod_type[m]}_FB{b+1}_{hl_corr_labels[h]}_{f+1}.mp4"
                    if randnames:
                        ntmp = rnames[0]
                        rnames = rnames[1:]
                        rname_tmp = f"Tin_Mod_File_{ntmp}.mp4"
                        file_key.append([fname_tmp, rname_tmp])

                    stim = []
                    for s in range(n_per_file):
                        dur_tmp = np.round(10 * (min(dur_range) + np.random.rand() * np.diff(dur_range)[0])) / 10
                        f0_tmp = int(np.round(min(f0_range) + np.random.rand() * np.diff(f0_range)[0]))
                        fband_db = hl_corr_temp_fbands * hl_corr_vals[h]
                        fband_mod = [b, b + 1]

                        stim_tmp = mod_ripple(f0_tmp, 0, dur_tmp, loud, dur_tmp * ramp_prop, 1,
                                            [1.5, 7.5], 8, fbands, fband_db.tolist(), fband_mod, mod_type[m], 0)
                        stim.extend(stim_tmp)

                    stim = np.array(stim)
                    maxabs = np.max(np.abs(stim))
                    if maxabs > 1:
                        stim = stim / maxabs
                        print('Downscaling to prevent clipping.')

                    if randnames:
                        sf.write(rname_tmp, stim, srate)
                    else:
                        sf.write(fname_tmp, stim, srate)

    if randnames:
        np.save(name_key_file, file_key)


def generate_hearing_tinnitus_estimation_stimuli() -> None:
    """
    Script to generate hearing and tinnitus estimation stimuli

    Generates pure tone (PT) and narrowband noise (NBN) stimuli for different
    hearing loss correction profiles.
    """
    rms = 0.01
    srate = 44100
    fhz = band_separation_frequencies()  # 1-16 kHz in 1/4 octave steps
    bw = 0.5  # Bandwidth (oct) of NBN stimuli
    ramp = 10
    stim_dur = 1
    isi = 1

    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 2.25, 0.25), [2, 2, 2, 2]]) / 2
    hl_corr_vals = [0, 15, 30, 45]  # Maximum (dB) correction difference between 8 kHz and 1 kHz
    hl_corr_labels = ['NH', 'MildHL', 'ModHL', 'SevHL']
    gen_ext = 'Hearing_Tinnitus_Estimation_Stimuli'
    type_ext = ['PT', 'NBN']

    tvec = np.arange(1/srate, stim_dur + 1/srate, 1/srate)
    zvec = np.zeros(int(stim_dur * srate))
    bw_mults = 2 ** np.array([-0.5, 0.5]) * bw

    for c in range(len(hl_corr_labels)):
        for t in range(2):  # 0 = PT, 1 = NBN
            stim_all = []
            for n in range(len(fhz)):
                if t == 0:  # Pure tone
                    stim_tmp = np.sin(2 * np.pi * fhz[n] * tvec)
                else:  # Narrowband noise
                    stim_tmp = farpn(srate, fhz[n] * bw_mults[0], fhz[n] * bw_mults[1], stim_dur)

                stim_tmp = stim_tmp * rms / np.std(stim_tmp)
                stim_tmp = wind_ramp(srate, ramp, stim_tmp)
                stim_tmp = stim_tmp * (10 ** ((hl_corr_temp[n] * hl_corr_vals[c]) / 20))
                stim_all.extend(stim_tmp)
                stim_all.extend(zvec)

            stim_all = np.array(stim_all)
            maxabs = np.max(np.abs(stim_all))
            if maxabs > 0.9:
                stim_all = stim_all / (maxabs * 1.1)
                print('Downscaling to prevent clipping.')

            filename = f"{gen_ext}_{type_ext[t]}_{hl_corr_labels[c]}.mp4"
            sf.write(filename, stim_all, srate)


def create_example_stimulus(f0: float = 150, mod_type: str = 'amp', duration: float = 4.0,
                          output_filename: str = 'example_tinnitus_stimulus.wav') -> np.ndarray:
    """
    Create a simple example tinnitus therapy stimulus

    Args:
        f0: fundamental frequency (Hz)
        mod_type: modulation type ('amp', 'phase', 'noise', 'none')
        duration: stimulus duration (seconds)
        output_filename: output audio file name

    Returns:
        generated stimulus array
    """
    # Set up frequency bands that can accommodate the fundamental frequency
    # Create bands that span from f0 to ~3*f0 to ensure harmonics fall within bands
    if f0 < 2000:
        # Low frequency - use original bands
        fbands = [[1000, 1414], [1414, 2000], [2000, 2828], [2828, 4000], [4000, 5657], [5657, 8000]]
        fband_mod = [2, 3]  # Modulate the 2-4 kHz bands
    else:
        # High frequency - create bands around f0 and its harmonics
        base_freq = f0 * 0.8  # Start slightly below f0
        fbands = []
        for i in range(6):
            low = base_freq * (1.4 ** i)
            high = base_freq * (1.4 ** (i + 1))
            fbands.append([low, high])
        fband_mod = [1, 2]  # Modulate bands containing f0 and 2*f0

    fbanddb = [0] * len(fbands)  # No hearing loss correction

    # Generate stimulus
    stimulus = mod_ripple(
        f0=f0,
        rand_phase=0,
        dur=duration,
        loud=0.1,
        ramp=1.0,
        tmr=1.0,
        smr=[1.5, 7.5],
        scyc=8,
        fbands=fbands,
        fbanddb=fbanddb,
        fband_mod=fband_mod,
        mod_type=mod_type,
        normfreq=0
    )

    # Save to file
    sf.write(output_filename, stimulus, 44100)
    print(f"Generated {mod_type} modulated stimulus: {output_filename}")

    return stimulus

def band_separation_frequencies() -> List[float]:
    return 1000 * (2 ** np.arange(0, 4.5, 0.5))

def frequency_bands(fb: List[float]) -> List[List[float]]:
    return [[fb[i], fb[i+1]] for i in range(8)]

def band_breakpoints() -> List[float]:
    """division points to assign frequencies to the band that contains closest to its center"""
    return 1000 * (2 ** (np.arange(3, 17, 2) / 4.0))

def frequency_to_band_index(frequency_hz: float) -> int:
    """
    Convert a tinnitus frequency to the appropriate frequency band pair index

    The modulation uses consecutive pairs of frequency bands. This function determines
    which pair index should be used such that fband_mod = [index, index+1] will
    modulate the appropriate bands containing the target frequency closest to the logarithmic center of the band.

    Args:
        frequency_hz: tinnitus frequency in Hz

    Returns:
        band pair index (0-6) where:
        - 0 → FB1-FB2 (1000-2000 Hz)
        - 1 → FB2-FB3 (1414-2828 Hz)
        - 2 → FB3-FB4 (2000-4000 Hz)
        - 3 → FB4-FB5 (2828-5657 Hz)
        - 4 → FB5-FB6 (4000-8000 Hz)
        - 5 → FB6-FB7 (5657-11314 Hz)
        - 6 → FB7-FB8 (8000-16000 Hz)

    Raises:
        ValueError: if frequency is outside the supported range
    """
    bandbreakpoints = band_breakpoints()
    for i in range(len(bandbreakpoints)):
        if frequency_hz <= bandbreakpoints[i]:
            break
    return i


def generate_cli_stimuli(frequency_hz: Optional[float] = None,
                        mod_band: Optional[str] = None,
                        mod_types: Optional[List[str]] = None,
                        hearing_profiles: Optional[List[str]] = None,
                        files_per_category: int = 1,
                        output_dir: str = ".",
                        file_prefix: str = "",
                        randnames: bool = False,
                        loud: float = 0.1,
                        duration_seconds: float = 3600,
                        generate_all: bool = False) -> None:
    """
    Generate tinnitus therapy stimuli based on CLI parameters

    Args:
        frequency_hz: target tinnitus frequency in Hz (None = all bands)
        mod_band: frequency band name (FB1-FB7) for modulation (None = all bands)
        mod_types: list of modulation types or None for all
        hearing_profiles: list of hearing profiles or None for ['NH']
        files_per_category: number of files per category
        output_dir: output directory
        file_prefix: prefix for generated filenames
        randnames: use random filenames
        loud: loudness scaling factor
        duration_seconds: duration of each generated file in seconds
        generate_all: generate all combinations (ignores other filters)
    """
    import os

    # Set defaults
    if mod_types is None:
        mod_types = ['noise', 'amp', 'phase']
    if hearing_profiles is None:
        hearing_profiles = ['NH']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if generate_all:
        print("Generating full experimental stimulus set...")
        generate_full_experiment_stimuli_updated(
            loud=loud,
            files_per_category=files_per_category,
            randnames=randnames,
            name_key_file=os.path.join(output_dir, 'Random_File_Names_Key')
        )
        return

    # Determine frequency bands to generate
    if frequency_hz is not None:
        try:
            band_index = frequency_to_band_index(frequency_hz)
            band_indices = [band_index]
            print(f"Target frequency: {frequency_hz} Hz → FB{band_index+1}-FB{band_index+2}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif mod_band is not None:
        # Convert band name (FB1-FB7) to index (0-6)
        band_index = int(mod_band[2:]) - 1  # Extract number and convert to 0-based index
        band_indices = [band_index]
        print(f"Target band: {mod_band} → FB{band_index+1}-FB{band_index+2}")
    else:
        band_indices = list(range(7))  # FB1-FB7 (0-6)
        print("Generating for all frequency bands (FB1-FB7)")

    print(f"Modulation types: {', '.join(mod_types)}")
    print(f"Hearing profiles: {', '.join(hearing_profiles)}")
    print(f"Files per category: {files_per_category}")
    print(f"Duration per file: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Output directory: {output_dir}")
    if file_prefix:
        print(f"File prefix: {file_prefix}")
    print()

    total_files = len(band_indices) * len(mod_types) * len(hearing_profiles) * files_per_category
    print(f"Will generate {total_files} files...")
    print()

    file_count = 0
    for band_idx in band_indices:
        for mod_type in mod_types:
            for hearing_profile in hearing_profiles:
                for file_num in range(1, files_per_category + 1):
                    try:
                        # Generate the stimulus
                        filepath = generate_single_experiment_stimulus(
                            mod_type=mod_type,
                            fband_index=band_idx,
                            hearing_profile=hearing_profile,
                            file_number=file_num,
                            loud=loud,
                            output_dir=output_dir,
                            randnames=randnames,
                            duration_seconds=duration_seconds
                        )

                        # Apply file prefix if specified
                        if file_prefix and not randnames:
                            old_path = filepath
                            filename = os.path.basename(filepath)
                            new_filename = f"{file_prefix}{filename}"
                            new_path = os.path.join(output_dir, new_filename)
                            os.rename(old_path, new_path)
                            print(f"  Renamed to: {new_filename}")

                        file_count += 1
                        print(f"Progress: {file_count}/{total_files} files completed")

                    except Exception as e:
                        print(f"Error generating {mod_type}_FB{band_idx+1}_{hearing_profile}_{file_num}: {e}")
                        continue

    print(f"\nCompleted! Generated {file_count} files in {output_dir}")


def main():
    """
    Command-line interface for tinnitus sound therapy stimulus generation
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate tinnitus sound therapy stimuli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                              # Generate all 84 files
  %(prog)s -f 3000                           # Generate for 3 kHz tinnitus
  %(prog)s --mod-band FB4                    # Generate for FB4-FB5 bands directly
  %(prog)s -f 4000 -m amp phase              # 4 kHz, amplitude and phase only
  %(prog)s -f 1500 --hearing-profile NH MildHL              # 1.5 kHz, normal and mild HL
  %(prog)s -f 8000 -n 3 -o therapy_files                 # 8 kHz, 3 files per category
  %(prog)s --mod-band FB2 -m noise --hearing-profile SevHL  # Direct band selection with modulation

Frequency Band Mapping:
  FB1: 1000-1414 Hz    FB2: 1414-2000 Hz    FB3: 2000-2828 Hz    FB4: 2828-4000 Hz
  FB5: 4000-5657 Hz    FB6: 5657-8000 Hz    FB7: 8000-11314 Hz
        """
    )

    # Main options
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all experimental stimuli (84 files, ~60GB)"
    )

    parser.add_argument(
        "-f", "--frequency",
        type=float,
        help="Tinnitus frequency in Hz (determines frequency band)"
    )

    parser.add_argument(
        "--mod-band",
        choices=["FB1", "FB2", "FB3", "FB4", "FB5", "FB6", "FB7"],
        help="Frequency band for modulation (alternative to --frequency)"
    )

    parser.add_argument(
        "-m", "--modulation",
        choices=["noise", "amp", "phase"],
        nargs="+",
        default=None,
        help="Modulation type(s) (default: all types)"
    )

    parser.add_argument(
        "--hearing-profile",
        choices=["NH", "MildHL", "ModHL", "SevHL"],
        nargs="+",
        default=["NH"],
        help="Hearing loss profile(s) (default: NH only)"
    )

    parser.add_argument(
        "-n", "--files-per-category",
        type=int,
        default=1,
        help="Number of files per category (default: 1)"
    )

    # Output options
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory (default: current directory)"
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for generated filenames"
    )

    parser.add_argument(
        "--random-names",
        action="store_true",
        help="Use random filenames instead of descriptive names"
    )

    parser.add_argument(
        "--loud",
        type=float,
        default=0.1,
        help="Loudness scaling factor (default: 0.1)"
    )

    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=3600,
        help="Duration of each file in seconds (default: 3600 = 1 hour)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if not args.all and args.frequency is None and args.mod_band is None:
        parser.error("Must specify either --all, -f/--frequency, or --mod-band")

    # Check for mutual exclusivity
    specified_options = sum([
        args.all,
        args.frequency is not None,
        args.mod_band is not None
    ])
    if specified_options > 1:
        parser.error("Cannot specify more than one of: --all, -f/--frequency, --mod-band")

    if args.files_per_category < 1:
        parser.error("files-per-category must be >= 1")

    if args.duration <= 0:
        parser.error("duration must be > 0")

    try:
        print("Tinnitus Sound Therapy - Python Implementation")
        print("=" * 50)

        generate_cli_stimuli(
            frequency_hz=args.frequency,
            mod_band=args.mod_band,
            mod_types=args.modulation,
            hearing_profiles=args.hearing_profile,
            files_per_category=args.files_per_category,
            output_dir=args.output_dir,
            file_prefix=args.prefix,
            randnames=args.random_names,
            loud=args.loud,
            duration_seconds=args.duration,
            generate_all=args.all
        )

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()