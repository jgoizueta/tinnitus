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

    Originally by Tim Griffiths, 2004
    Part of Pitch Stimulus Design Toolbox
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
    x = sig / (10 * np.std(sig))

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
        fbands: lower and upper limits (Hz) of specified frequency bands
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
    fn = []
    fmod = []
    fintens = []
    band_ind = [[] for _ in range(len(fbands))]
    ind_tot = 0
    nf = len(fbands)

    for f in range(nf):
        frange_tmp = fbands[f]
        fntmp = list(range(int(np.ceil(frange_tmp[0]/f0)), int(np.floor((frange_tmp[1]-1)/f0)) + 1))
        fn.extend(fntmp)  # Number of harmonics

        if f in fband_mod:
            fmod.extend([1] * len(fntmp))  # Yes or no to harmonic modulation
        else:
            fmod.extend([0] * len(fntmp))

        fintens.extend([10**(fbanddb[f]/20)] * len(fntmp))  # Relative intensity of harmonics
        nind_tmp = len(fntmp)
        band_ind[f] = list(range(ind_tot, ind_tot + nind_tmp))  # Indices of harmonics belonging to each frequency band
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
        noise_tmp = noise_tmp * stmp_std / noise_std  # Scale to same RMS as harmonic band being replaced
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

    fb = 1000 * (2 ** np.arange(0, 4.5, 0.5))
    fbands = [[fb[i], fb[i+1]] for i in range(8)]  # Frequency bands to apply modulation
    fhz = 1000 * (2 ** np.arange(0, 4.25, 0.25))
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
    fhz = 1000 * (2 ** np.arange(0, 4.25, 0.25))  # 1-16 kHz in 1/4 octave steps
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
    # Set up frequency bands (example using 2-4 kHz band)
    fbands = [[1000, 1414], [1414, 2000], [2000, 2828], [2828, 4000], [4000, 5657], [5657, 8000]]
    fbanddb = [0, 0, 0, 0, 0, 0]  # No hearing loss correction
    fband_mod = [2, 3]  # Modulate the 2-4 kHz bands

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


def main():
    """
    Example usage of the tinnitus sound therapy functions
    """
    print("Tinnitus Sound Therapy - Python Implementation")
    print("=" * 50)

    # Create example stimuli with different modulation types
    print("Creating example stimuli...")

    create_example_stimulus(f0=150, mod_type='amp', output_filename='amplitude_modulation_example.wav')
    create_example_stimulus(f0=150, mod_type='phase', output_filename='phase_modulation_example.wav')
    create_example_stimulus(f0=150, mod_type='noise', output_filename='noise_modulation_example.wav')

    print("\nExample stimuli created successfully!")
    print("\nTo generate full experimental stimuli (WARNING: creates many files):")
    print("generate_full_experiment_stimuli(files_per_category=1)")
    print("\nTo generate hearing estimation stimuli:")
    print("generate_hearing_tinnitus_estimation_stimuli()")


if __name__ == "__main__":
    main()