import numpy as np
import warnings


def solve_with_nan_handling(C, Z):
    """
    Solve the least-squares problem C * x = Z with MATLAB-like behavior for NaN values.
    If NaNs are present in C or Z, return a vector of zeros with appropriate dimensions.
    """
    # Check for NaN in C or Z
    if np.isnan(C).any() or np.isnan(Z).any():
        # MATLAB returns zeros in this scenario
        num_cols = C.shape[1] if C.ndim > 1 else 1
        return np.zeros((num_cols,))

    # Perform least-squares solving
    try:
        solution, _, _, _ = np.linalg.lstsq(C, Z, rcond=None)
        return solution
    except np.linalg.LinAlgError:
        return np.zeros((C.shape[1],))

def EMEP(xps, trm, kx, Ss, pidirs, miter, displ):
    szd = xps.shape[0]
    freqs = xps.shape[2]
    ddirs = trm.shape[2]

    ddir = abs(pidirs[1] - pidirs[0])
    pi = np.pi

    if displ < 2:
        warnings.simplefilter("ignore")

    Co = np.real(xps)
    Quad = -np.imag(xps)

    sigCo = np.zeros_like(Co)
    sigQuad = np.zeros_like(Quad)
    xpsx = np.zeros_like(Co)
    for ff in range(freqs):
        xpsx[:,:,ff] = np.outer(np.real(np.diag(xps[:,:,ff])),
                                  np.real(np.diag(xps[:,:,ff]).T))
        sigCo[:,:,ff] = np.sqrt(0.5 * (xpsx[:,:,ff] + Co[:,:,ff]**2 - Quad[:,:,ff]**2))
        sigQuad[:,:,ff] = np.sqrt(0.5 * (xpsx[:,:,ff] - Co[:,:,ff]**2 + Quad[:,:,ff]**2))

    S = np.zeros((freqs, ddirs))

    phi = np.zeros((szd+2, freqs))
    H = np.zeros((ddirs, szd+2, freqs))
    for ff in range(freqs):
        index = 0

        for m in range(szd):
            for n in range(m, szd):
                expx = np.exp(-1j * kx[m, n, ff, :ddirs])
                Hh = trm[m, ff, :ddirs]
                Hhs = np.conj(trm[n, ff, :ddirs])
                Htemp = Hh * Hhs * expx

                if Htemp[0] != Htemp[1]:
                    phi[index,ff] = (np.real(xps[m, n, ff]) / (sigCo[m, n, ff] * Ss[0, ff]))
                    H[0:ddirs,index,ff] = np.real(Htemp) / sigCo[m, n, ff]
                    index += 1

                    if kx[m, n, 0, 0] + kx[m, n, 0, 1] != 0:
                        phi[index,ff] = (np.imag(xps[m, n, ff]) / (sigQuad[m, n, ff] * Ss[0, ff]))
                        H[0:ddirs,index,ff] = np.imag(Htemp) / sigQuad[m, n, ff]
                        index += 1

    M = index

    cosnt = np.zeros((ddirs, M, M // 2 + 1))
    sinnt = np.zeros((ddirs, M, M // 2 + 1))
    for eni in range(1, M // 2 + 2):
        cosnt[:, :, eni - 1] = np.cos(eni * pidirs)[:, None]
        sinnt[:, :, eni - 1] = np.sin(eni * pidirs)[:, None]
    cosn = np.cos(np.arange(1, M // 2 + 2)[:, None] * pidirs)
    sinn = np.sin(np.arange(1, M // 2 + 2)[:, None] * pidirs)

    for ff in range(freqs):
        if displ >= 1:
            print(f"Calculating for frequency {ff + 1} of {freqs}")

        Hi = H[0:ddirs,0:M,ff]
        Phione = np.outer(np.ones_like(pidirs), phi[:M,ff])

        keepgoing = True
        n = 0
        AIC = []

        a1held, b1held = [], []
        while keepgoing:
            n += 1

            if n <= M // 2 + 1:
                if displ > 0:
                    print(f"Model: {n}")

                a1, b1 = np.zeros(n), np.zeros(n)
                a2, b2 = np.ones(n) * 100, np.ones(n) * 100
                count = 0
                rlx = 1.0

                while np.max(np.abs(a2)) > 0.01 or np.max(np.abs(b2)) > 0.01:
                    count += 1
                    Fn = (a1 @ cosn[:n, :] + b1 @ sinn[:n, :]).T

                    Fnexp = np.exp(Fn)[:, None] * np.ones((1, M))
                    PhiHF = (Phione - Hi) * Fnexp
                    Z = np.sum(PhiHF, axis=0) / np.sum(Fnexp, axis=0)

                    X = np.zeros((n, M))
                    Y = np.zeros((n, M))
                    for eni in range(n):
                        X[eni,0:M] = Z * ((
                            np.sum(Fnexp * cosnt[:,:,eni], axis=0) / \
                                np.sum(Fnexp, axis=0)
                        ) - (
                            np.sum(PhiHF * cosnt[:,:,eni], axis=0) / \
                                np.sum(PhiHF, axis=0)
                        ))
                        Y[eni,0:M] = Z * ((
                            np.sum(Fnexp * sinnt[:,:,eni], axis=0) / \
                                np.sum(Fnexp, axis=0)
                        ) - (
                            np.sum(PhiHF * sinnt[:,:,eni], axis=0) / \
                                np.sum(PhiHF, axis=0)
                        ))

                    C = np.hstack((X.T, Y.T))

                    out = solve_with_nan_handling(C, Z)

                    a2old, b2old = a2.copy(), b2.copy()
                    a2, b2 = out[:n], out[n:2*n]

                    if (
                        np.sum(np.abs(a2) - np.abs(a2old) > 100) > 0
                        or np.sum(np.abs(b2) - np.abs(b2old) > 100) > 0
                        or count > miter
                    ):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            if displ == 2:
                                print(f"Relaxing computation...factor: {rlx:.4f}")

                            count = 0
                            a1[:n], b1[:n] = 0.0, 0.0
                        else:
                            if displ == 2:
                                print("Computation fully relaxed...bailing out")
                            keepgoing = False
                            break
                    else:
                        a1 = a1 + rlx*a2
                        b1 = b1 + rlx*b2

                error = Z - a2 @ X - b2 @ Y
                AIC.append(M * (np.log(2 * pi * np.var(error)) + 1) + 4 * n + 2)

                if n > 1 and (AIC[-1] > AIC[-2] or np.isnan(AIC[-1])):
                    keepgoing = False

                a1held.append(a1[:n])
                b1held.append(b1[:n])
                best = n

                if not keepgoing:
                    if n > 1:
                        max_len = max(len(arr) for arr in a1held)
                        padded_a1held = [np.pad(
                            arr, (0, max_len - len(arr))) for arr in a1held]
                        padded_b1held = [np.pad(
                            arr, (0, max_len - len(arr))) for arr in b1held]
                        a1 = np.array(padded_a1held)[n - 2, :n - 1]
                        b1 = np.array(padded_b1held)[n - 2, :n - 1]
                        best = n - 1
                    else:
                        a1 = np.zeros(1)
                        b1 = np.zeros(1)

            else:
                keepgoing = False

        if displ == 2:
            print(f"Best: {best}")

        G = np.exp(a1@cosn[:best, :] + \
                   b1@sinn[:best, :]).T
        SG = G / (np.sum(G) * ddir)
        S[ff, 0:ddirs] = Ss[0, ff] * SG.T

    return S
