#!/usr/bin/env python3
"""
Hearing Assessment Stimuli Generator

Python port of Hearing_Tin_Stim_Generation.m
Generates hearing and tinnitus frequency estimation stimuli

This script generates test stimuli for hearing assessment with two types:
- PT (Pure Tone): Sine wave signals at specific frequencies
- NBN (Narrowband Noise): Filtered noise centered at specific frequencies

The stimuli are generated for 4 hearing loss correction profiles:
- NH (Normal Hearing): 0 dB correction
- MildHL (Mild Hearing Loss): 15 dB max correction
- ModHL (Moderate Hearing Loss): 30 dB max correction
- SevHL (Severe Hearing Loss): 45 dB max correction

Each file contains 17 frequencies from 1-16 kHz in 1/4 octave steps.

Original MatLab code by the tinnitus research team.
Python port for standalone hearing assessment stimulus generation.
"""

import numpy as np
import soundfile as sf
from typing import List, Optional
import argparse
import os


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


def generate_hearing_assessment_stimuli(output_dir: str = ".",
                                       file_format: str = "wav",
                                       verbose: bool = True) -> None:
    """
    Generate hearing and tinnitus frequency estimation stimuli

    This function generates test stimuli exactly matching the MatLab implementation.
    Creates PT (Pure Tone) and NBN (Narrowband Noise) stimuli for different
    hearing loss correction profiles.

    Args:
        output_dir: directory to save generated files (default: current directory)
        file_format: audio file format - "wav" or "mp4" (default: "wav")
        verbose: whether to print progress information (default: True)
    """
    # Parameters from MatLab implementation
    rms = 0.01
    srate = 44100
    fhz = 1000 * (2 ** np.arange(0, 4.25, 0.25))  # 1-16 kHz in 1/4 octave steps
    bw = 0.5  # Bandwidth (oct) of NBN stimuli
    ramp = 10  # Ramp duration (ms)
    stim_dur = 1  # Duration of each stimulus (s)
    isi = 1  # Inter-stimulus interval (s)

    # Hearing loss correction template and values
    hl_corr_temp = np.concatenate([[0, 0, 0, 0], np.arange(0, 2.25, 0.25), [2, 2, 2, 2]]) / 2
    hl_corr_vals = [0, 15, 30, 45]  # Maximum (dB) correction difference between 8 kHz and 1 kHz
    hl_corr_labels = ['NH', 'MildHL', 'ModHL', 'SevHL']

    gen_ext = 'Hearing_Tinnitus_Estimation_Stimuli'
    type_ext = ['PT', 'NBN']

    # Create time vectors
    tvec = np.arange(1/srate, stim_dur + 1/srate, 1/srate)
    zvec = np.zeros(int(stim_dur * srate))  # Silent interval
    bw_mults = 2 ** (np.array([-0.5, 0.5]) * bw)

    # Validate file format and adjust if needed
    supported_formats = ['wav', 'flac', 'ogg']
    if file_format == 'mp4':
        if verbose:
            print("Warning: MP4 format not supported by Python soundfile library.")
            print("Converting to WAV format for compatibility.")
        file_format = 'wav'
    elif file_format not in supported_formats:
        if verbose:
            print(f"Warning: Format '{file_format}' may not be supported.")
            print(f"Supported formats: {', '.join(supported_formats)}")
            print("Proceeding with requested format...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("Generating Hearing Assessment Stimuli")
        print("=" * 50)
        print(f"Frequencies: {len(fhz)} steps from {fhz[0]:.0f} to {fhz[-1]:.0f} Hz")
        print(f"Hearing profiles: {len(hl_corr_labels)} ({', '.join(hl_corr_labels)})")
        print(f"Stimulus types: {len(type_ext)} ({', '.join(type_ext)})")
        print(f"Total files to generate: {len(hl_corr_labels) * len(type_ext)}")
        print(f"Output directory: {output_dir}")
        print(f"File format: {file_format}")
        print()

    file_count = 0
    total_files = len(hl_corr_labels) * len(type_ext)

    for c in range(len(hl_corr_labels)):
        for t in range(len(type_ext)):  # 0 = PT, 1 = NBN
            stim_all = []

            if verbose:
                print(f"Generating {type_ext[t]} stimuli for {hl_corr_labels[c]} profile...")

            for n in range(len(fhz)):
                if t == 0:  # Pure tone (PT)
                    stim_tmp = np.sin(2 * np.pi * fhz[n] * tvec)
                else:  # Narrowband noise (NBN)
                    stim_tmp = farpn(srate, fhz[n] * bw_mults[0], fhz[n] * bw_mults[1], stim_dur)

                # Normalize to target RMS
                stim_tmp = stim_tmp * rms / np.std(stim_tmp)

                # Apply ramping
                stim_tmp = wind_ramp(srate, ramp, stim_tmp)

                # Apply hearing loss correction
                correction_db = hl_corr_temp[n] * hl_corr_vals[c]
                stim_tmp = stim_tmp * (10 ** (correction_db / 20))

                # Add stimulus and silent interval
                stim_all.extend(stim_tmp)
                stim_all.extend(zvec)

            # Convert to numpy array and check for clipping
            stim_all = np.array(stim_all)
            maxabs = np.max(np.abs(stim_all))
            if maxabs > 0.9:
                stim_all = stim_all / (maxabs * 1.1)
                if verbose:
                    print('  Downscaling to prevent clipping.')

            # Generate filename and save
            filename = f"{gen_ext}_{type_ext[t]}_{hl_corr_labels[c]}.{file_format}"
            filepath = os.path.join(output_dir, filename)

            sf.write(filepath, stim_all, srate)

            file_count += 1
            if verbose:
                duration_min = len(stim_all) / srate / 60
                print(f"  Generated: {filename} ({duration_min:.1f} minutes, {file_count}/{total_files})")

    if verbose:
        print()
        print("=" * 50)
        print("Hearing assessment stimuli generation completed!")
        print(f"Generated {file_count} files in {output_dir}")
        print()
        print("File description:")
        print("- PT files: Pure tone stimuli (sine waves)")
        print("- NBN files: Narrowband noise stimuli")
        print("- NH: Normal Hearing (no correction)")
        print("- MildHL: Mild Hearing Loss (15 dB max correction)")
        print("- ModHL: Moderate Hearing Loss (30 dB max correction)")
        print("- SevHL: Severe Hearing Loss (45 dB max correction)")
        print()
        print("Each file contains 17 frequencies from 1-16 kHz in 1/4 octave steps")
        print("with 1-second stimuli separated by 1-second silent intervals.")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Generate hearing assessment stimuli for tinnitus frequency estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Generate in current directory
  %(prog)s -o hearing_stimuli       # Generate in specific directory
  %(prog)s -f flac --quiet          # Generate FLAC files without progress
  %(prog)s --help                   # Show this help message

Output files:
  - Hearing_Tinnitus_Estimation_Stimuli_PT_NH.wav     (Pure tones, Normal Hearing)
  - Hearing_Tinnitus_Estimation_Stimuli_NBN_NH.wav    (Narrowband noise, Normal Hearing)
  - Hearing_Tinnitus_Estimation_Stimuli_PT_MildHL.wav (Pure tones, Mild Hearing Loss)
  - ... (8 files total)

Each file contains 17 test frequencies from 1-16 kHz in 1/4 octave steps.
        """
    )

    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for generated files (default: current directory)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["wav", "flac", "ogg", "mp4"],
        default="wav",
        help="Audio file format (default: wav). Note: mp4 will be converted to wav due to library limitations."
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        generate_hearing_assessment_stimuli(
            output_dir=args.output_dir,
            file_format=args.format,
            verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())