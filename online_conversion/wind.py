import numpy as np

def wind(srate, wdms, x):
    npts = len(x)
    wds = round(2 * wdms / 1000 * srate)
    if wds % 2 != 0:
        wds += 1
    w = np.linspace(-1 * (np.pi / 2), 1.5 * np.pi, wds)
    w = (np.sin(w) + 1) / 2
    x[:round(wds / 2)] = x[:round(wds / 2)] * w[:round(wds / 2)]
    if srate == 48828:
        x[npts - round(wds / 2):npts] = x[npts - round(wds / 2):npts] * w[round(wds / 2):wds]
    else:
        x[npts - round(wds / 2):npts] = x[npts - round(wds / 2):npts] * w[round(wds / 2) + 1:wds]
    return x
