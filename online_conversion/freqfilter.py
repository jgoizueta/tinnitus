import numpy as np

def freqfilter(fs, f1, f2, s1):
    s1 = np.array(s1)
    if len(s1) % 2 != 0:
        s1 = s1[:-1]
    npts = len(s1)
    fbin = fs / npts
    f = np.fft.fft(s1)
    f[:int(f1 // fbin)] = 0
    f[int((f2 // fbin) + 1):npts - int(f2 // fbin)] = 0
    f[npts - int(f1 // fbin):] = 0
    s2 = np.real(np.fft.ifft(f))
    return s2
