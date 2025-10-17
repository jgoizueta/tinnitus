"""
Simple test script to verify the Python implementation of tinnitus sound therapy functions
"""

import numpy as np
from tinnitus_sound_therapy import *

def test_farpn():
    """Test the farpn function"""
    print("Testing farpn function...")

    # Generate noise
    noise = farpn(44100, 1000, 2000, 1.0)

    # Check basic properties
    assert len(noise) > 0, "Noise should not be empty"
    assert np.abs(np.std(noise) - 0.1) < 0.01, f"RMS should be ~0.1, got {np.std(noise)}"

    # Check frequency content using FFT
    fft_noise = np.fft.fft(noise)
    freqs = np.fft.fftfreq(len(noise), 1/44100)
    power = np.abs(fft_noise)**2

    # Check that most power is in the specified frequency range
    in_band = np.sum(power[(freqs >= 1000) & (freqs <= 2000)])
    total = np.sum(power[freqs >= 0])
    ratio = in_band / total

    print(f"  - Generated {len(noise)} samples")
    print(f"  - RMS: {np.std(noise):.4f}")
    print(f"  - Power in band (1-2 kHz): {ratio:.2%}")
    assert ratio > 0.8, f"Most power should be in specified band, got {ratio:.2%}"
    print("  ✓ farpn test passed")

def test_wind_ramp():
    """Test the wind_ramp function"""
    print("Testing wind_ramp function...")

    # Create a step signal
    signal = np.ones(4410)  # 0.1 second at 44100 Hz

    # Apply ramping
    ramped = wind_ramp(44100, 10, signal)  # 10ms ramps

    # Check that signal is modified at edges
    assert ramped[0] < signal[0], "Signal should be attenuated at start"
    assert ramped[-1] < signal[-1], "Signal should be attenuated at end"
    assert np.abs(ramped[len(ramped)//2] - signal[len(signal)//2]) < 1e-10, "Middle should be unchanged"

    print(f"  - Original signal range: [{np.min(signal):.2f}, {np.max(signal):.2f}]")
    print(f"  - Ramped signal range: [{np.min(ramped):.2f}, {np.max(ramped):.2f}]")
    print(f"  - Start value: {ramped[0]:.4f}")
    print(f"  - End value: {ramped[-1]:.4f}")
    print("  ✓ wind_ramp test passed")

def test_mod_ripple():
    """Test the mod_ripple function"""
    print("Testing mod_ripple function...")

    # Set up test parameters
    fbands = [[1000, 1414], [1414, 2000], [2000, 2828], [2828, 4000]]
    fbanddb = [0, 0, 0, 0]
    fband_mod = [1, 2]

    # Test different modulation types
    for mod_type in ['amp', 'phase', 'noise', 'none']:
        stim = mod_ripple(
            f0=200, rand_phase=0, dur=0.5, loud=0.1, ramp=0.05,
            tmr=1.0, smr=[1.5, 7.5], scyc=8,
            fbands=fbands, fbanddb=fbanddb, fband_mod=fband_mod,
            mod_type=mod_type, normfreq=0
        )

        assert len(stim) > 0, f"Stimulus should not be empty for {mod_type}"
        assert np.max(np.abs(stim)) <= 1.0, f"Stimulus should not clip for {mod_type}"
        print(f"  - {mod_type}: {len(stim)} samples, max: {np.max(np.abs(stim)):.4f}")

    print("  ✓ mod_ripple test passed")

def test_stimulus_generation():
    """Test stimulus generation functions"""
    print("Testing stimulus generation...")

    # Test create_example_stimulus
    stim = create_example_stimulus(f0=150, mod_type='amp', duration=0.1,
                                  output_filename='test_output.wav')

    assert len(stim) > 0, "Generated stimulus should not be empty"
    assert np.max(np.abs(stim)) <= 1.0, "Generated stimulus should not clip"

    print(f"  - Generated stimulus: {len(stim)} samples")
    print(f"  - Max amplitude: {np.max(np.abs(stim)):.4f}")
    print("  ✓ stimulus generation test passed")

if __name__ == "__main__":
    print("Running tests for tinnitus sound therapy implementation...")
    print("=" * 60)

    try:
        test_farpn()
        print()
        test_wind_ramp()
        print()
        test_mod_ripple()
        print()
        test_stimulus_generation()
        print()
        print("=" * 60)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise