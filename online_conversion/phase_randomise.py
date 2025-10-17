import numpy as np

def phase_randomise(stim_in, srate, time_bin):
    time_bin = 2 * round(time_bin * srate / 2000)
    stim_in = stim_in[:len(stim_in) - (len(stim_in) % time_bin)]
    stim_out_a = np.zeros_like(stim_in)
    stim_out_b = np.zeros_like(stim_in)
    nbins = len(stim_in) // time_bin
    hanwin = np.hanning(time_bin)
    hanstart = np.concatenate((np.ones(time_bin // 2), hanwin[time_bin // 2:]))
    hanend = hanstart[::-1]

    for cur_bin in range(1, int(np.floor(nbins)) + 1):
        start_idx = (cur_bin - 1) * time_bin
        end_idx = cur_bin * time_bin
        bin_f = np.fft.fft(stim_in[start_idx:end_idx])
        bin_f = np.abs(bin_f)
        bin_phase = 2 * np.pi * np.random.rand(time_bin // 2 - 1)
        bin_phase = np.concatenate(([0], bin_phase, [0], -np.flip(bin_phase)))
        complex_phase = np.exp(1j * bin_phase)
        if cur_bin == 1:
            stim_out_a[start_idx:end_idx] = hanstart * np.real(np.fft.ifft(bin_f * complex_phase))
        elif cur_bin == int(np.floor(nbins)):
            stim_out_a[start_idx:end_idx] = hanend * np.real(np.fft.ifft(bin_f * complex_phase))
        else:
            stim_out_a[start_idx:end_idx] = hanwin * np.real(np.fft.ifft(bin_f * complex_phase))

    for cur_bin in range(1, int(np.floor(nbins - 1)) + 1):
        start_idx = int((cur_bin - 0.5) * time_bin)
        end_idx = int((cur_bin + 0.5) * time_bin)
        bin_f = np.fft.fft(stim_in[start_idx:end_idx])
        bin_f = np.abs(bin_f)
        bin_phase = 2 * np.pi * np.random.rand(time_bin // 2 - 1)
        bin_phase = np.concatenate(([0], bin_phase, [0], -np.flip(bin_phase)))
        complex_phase = np.exp(1j * bin_phase)
        stim_out_b[start_idx:end_idx] = hanwin * np.real(np.fft.ifft(bin_f * complex_phase))

    stim_out = stim_out_a + stim_out_b
    return stim_out
