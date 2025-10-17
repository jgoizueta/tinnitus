import numpy as np

def farpn(fs, lo, hi, dur):
    t = np.arange(0, dur, 1/fs)
    npts = (len(t) // 2) * 2
    fbin = fs / npts

    mag = np.ones(npts // 2)
    mag[:int(np.floor(lo / fbin))] = 0
    mag[int(np.ceil(hi / fbin)):] = 0

    phase = 2 * np.pi * np.random.rand(npts // 2 - 1)
    allphase = np.concatenate(([0], phase, [0], -np.flip(phase)))
    allmag = np.concatenate((mag, np.flip(mag)))

    rect = allmag * np.exp(1j * allphase)
    sig = np.real(np.fft.ifft(rect))

    x = sig / (10 * np.std(sig))  # set rms value to 0.1
    return x
