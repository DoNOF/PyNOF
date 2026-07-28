import pynof
import numpy as np
from numba import prange, njit, jit
from scipy.optimize import minimize
import math

try:
    import cupy as cp
except:
    pass


# CJCKD5
# @njit(parallel=True, cache=True)
def CJCKD4(n, no1, ndoc, nsoc, nbeta, nalpha, ndns, ncwo, MSpin):
    """PNOF4 coefficients C^J and C^K that multiply J and K integrals

    E = 2\sum_p n_pH_p + \sum_{pq}' C^J_{pq}J_{qp} - \sum_{pq}' C^K_{pq}K_{qp}

    T = (1-S_F)/S_F
    R_pq = h_p n_q/S_F; p \in occ, q \in vir

    \Delta_{pq} = \begin{cases}
                 h_p h_q   & if p \in occ, q \in occ\\
                 T h_p n_q & if p \in occ, q \in vir\\
                 T n_p h_q & if p \in vir, q \in occ\\
                 n_p n_q   & if p \in vir, q \in vir\\
               \end{cases}

    \Pi_{pq} = \begin{cases}
                -\sqrt{h_p h_q}                      & if p \in occ, q \in occ\\
                -\sqrt{R_pq} \sqrt{n_p - n_q + R_pq} & if p \in occ, q \in vir\\
                -\sqrt{R_qp} \sqrt{n_q - n_p + R_qp} & if p \in vir, q \in occ\\
                 \sqrt{n_p n_q}                    & if p \in vir, q \in vir\\
               \end{cases}

    C^J = 2(n_pn_q - Delta_pq)
    C^K = n_pn_q - Delta_pq - Pi_pq
    """

    nbf5 = np.shape(n)[0]

    h = 1 - n
    S_F = np.sum(h[:ndoc])

    T = (1 - S_F) / S_F
    R = np.outer(h[:ndoc], n[ndoc:]) / S_F

    Delta = np.zeros((nbf5, nbf5))
    Delta[:ndoc, :ndoc] = np.outer(h[:ndoc], h[:ndoc])
    Delta[:ndoc, ndoc:] = (1 - S_F) / S_F * np.outer(h[:ndoc], n[ndoc:])
    Delta[ndoc:, :ndoc] = (1 - S_F) / S_F * np.outer(n[ndoc:], h[:ndoc])
    Delta[ndoc:, ndoc:] = np.outer(n[ndoc:], n[ndoc:])

    Pi = np.zeros((nbf5, nbf5))
    Pi[:ndoc, :ndoc] = -np.sqrt(np.outer(h[:ndoc], h[:ndoc]))
    Pi[:ndoc, ndoc:] = -np.sqrt(np.outer(h[:ndoc], n[ndoc:]) / S_F) * np.sqrt(np.subtract.outer(n[:ndoc], n[ndoc:]) + R)
    Pi[ndoc:, :ndoc] = -np.sqrt(np.outer(n[ndoc:], h[:ndoc]) / S_F) * np.sqrt(
        -np.subtract.outer(n[ndoc:], n[:ndoc]) + R.T
    )
    Pi[ndoc:, ndoc:] = np.sqrt(np.outer(n[ndoc:], n[ndoc:]))

    cj12 = 2 * (np.outer(n, n) - Delta)
    ck12 = np.outer(n, n) - Delta - Pi

    return cj12, ck12


# @njit(parallel=True, cache=True)
def der_CJCKD4(n, ista, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo):
    """PNOF4 coefficients DC^J and DC^K that multiply J and K integrals

    dE/dgamma_g = 2\sum_p dn_p/dgamma_g H_p 
                  +\sum_{pq} dC^J_{pq}/dgamma_g J_{qp}
                  -\sum_{pq} dC^K_{pq}/dgamma_g K_{qp}

    T = (1-S_F)/S_F
    R_pq = h_p n_q/S_F; p \in occ, q \in vir

    dS_F/dx_g = - sum_p^F dn_p/dx_g
    dT/dx_g = -(dS_F/dx_g)/S_F^2
    dR_pq/dx_g = [(dh_p/dx_g n_q + h_p dn_q/dx_g) S_F - (h_p n_q) dS_F/dx_g ] / S_F^2

    dDelta_{pq}/dx_g = \begin{cases}
                 dh_p/dx_g h_q + h_p dh_q/dx_g                       & if p \in occ, q \in occ\\
                 dT/dx_g h_p n_q + T dh_p/dx_g n_q + T h_p dn_q/dx_g & if p \in occ, q \in vir\\
                 dT/dx_g n_p h_q + T dn_p/dx_g h_q + T n_p dh_q/dx_g & if p \in vir, q \in occ\\
                 dn_p/dx_g n_q + n_p dn_q/dx_g                       & if p \in vir, q \in vir\\
               \end{cases}

    dPi_{pq}/dx_g = \begin{cases}
      -1/2*(1/\sqrt{h_p} dh_p/dx_g \sqrt{h_q} + \sqrt{h_p} 1/\sqrt{h_q} dh_q/dx_g )  & if p \in occ, q \in occ\\
      -((1/2\sqrt{R_pq}) dR_pq/dx_g \sqrt{n_p - n_q + R_pq} + \sqrt{R_pq}(dn_p/dx_g - dn_q/dx_g + dR_pq/dx_g)/(2\sqrt{n_q-n_p+R_pq})) & if p \in occ, q \in vir\\
      -((1/2\sqrt{R_qp}) dR_qp/dx_g \sqrt{n_q - n_p + R_pq} + \sqrt{R_qp}(dn_q/dx_g - dn_p/dx_g + dR_qp/dx_g)/(2\sqrt{n_p-n_q+R_qp})) & if p \in vir, q \in occ\\
      1/2*(1/\sqrt{n_p} dn_p/dx_g \sqrt{n_q} + \sqrt{n_p} 1/\sqrt{n_q} dn_q/dx_g )  & if p \in vir, q \in vir\\
      \end{cases}

    """

    nbf5 = np.shape(n)[0]

    h = 1 - n
    dh_dgamma = -dn_dgamma

    S_F = max(np.sum(h[:ndoc]), 1e-7)
    D_S_F = np.zeros(nv)
    for k in prange(nv):
        D_S_F[k] = np.sum(dh_dgamma[:ndoc, k])

    T = (1 - S_F) / S_F
    D_T = np.zeros(nv)
    for k in prange(nv):
        D_T[k] = -D_S_F[k] / S_F**2

    R = np.outer(h[:ndoc], n[ndoc:]) / S_F
    D_R = np.zeros((ndoc, nbf5 - ndoc, nv))
    for k in prange(nv):
        D_R[:, :, k] = (
            (np.outer(dh_dgamma[:ndoc, k], n[ndoc:]) + np.outer(h[:ndoc], dn_dgamma[ndoc:, k])) * S_F
            - np.outer(h[:ndoc], n[ndoc:]) * D_S_F[k]
        ) / S_F**2

    D_Delta = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        D_Delta[:ndoc, :ndoc, k] = np.outer(dh_dgamma[:ndoc, k], h[:ndoc])
        D_Delta[:ndoc, ndoc:, k] = D_T[k] * np.outer(h[:ndoc], n[ndoc:]) + T * np.outer(dh_dgamma[:ndoc, k], n[ndoc:])
        D_Delta[ndoc:, :ndoc, k] = T * np.outer(dn_dgamma[ndoc:, k], h[:ndoc])
        D_Delta[ndoc:, ndoc:, k] = np.outer(dn_dgamma[ndoc:, k], n[ndoc:])

    D_Pi = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        D_Pi[:ndoc, :ndoc, k] = -np.outer(
            1 / np.maximum(2 * np.sqrt(h[:ndoc]), 1e-7) * dh_dgamma[:ndoc, k], np.sqrt(h[:ndoc])
        )
        D_Pi[:ndoc, ndoc:, k] = -(
            1 / np.maximum(2 * np.sqrt(R), 1e-7) * D_R[:, :, k] * np.sqrt(np.subtract.outer(n[:ndoc], n[ndoc:]) + R)
            + np.sqrt(R)
            * (np.subtract.outer(dn_dgamma[:ndoc, k], dn_dgamma[ndoc:, k]) + D_R[:, :, k])
            / np.maximum(2 * np.sqrt(np.subtract.outer(n[:ndoc], n[ndoc:]) + R), 1e-7)
        )
        D_Pi[ndoc:, ndoc:, k] = np.outer(
            1 / np.maximum(2 * np.sqrt(n[ndoc:]), 1e-7) * dn_dgamma[ndoc:, k], np.sqrt(n[ndoc:])
        )

    Dcj12r = np.zeros((nbf5, nbf5, nv))
    Dck12r = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        Dcj12r[:, :, k] = 2 * (np.outer(dn_dgamma[:, k], n) - D_Delta[:, :, k])
        Dck12r[:, :, k] = np.outer(dn_dgamma[:, k], n) - D_Delta[:, :, k] - D_Pi[:, :, k]

    return Dcj12r, Dck12r


# CJCKD5
@njit(parallel=True, cache=True)
def CJCKD5(n, no1, ndoc, nsoc, nbeta, nalpha, ndns, ncwo, MSpin):
    """PNOF5 coefficients C^J and C^K that multiply J and K integrals

    E = 2\sum_p n_pH_p + \sum_{pq} C^J_{pq}J_{qp} - \sum_{pq} C^K_{pq}K_{qp}

    C^J_{pq} = \begin{cases}
                 2 n_p n_q & if p \in \Omega_g, q \in \Omega_f,  g \neq g\\
                 0         & otherwise
               \end{cases}

    C^K_{pq} = \begin{cases}
                 n_p n_q        & if p \in \Omega_g, q \in \Omega_f,  g \neq g\\
                 \sqrt{n_p n_q} & if p,q \in \Omega_g ( (p \leq F, q > F) or (p > F, q \leq F) )\\
                -\sqrt{n_p n_q} & if p \neq q \in \Omega_g p,q > F \\
                 0              & otherwise
               \end{cases}
    """

    # Interpair Electron correlation #
    cj12 = 2 * np.outer(n, n)
    ck12 = np.outer(n, n)

    # Intrapair Electron Correlation

    if MSpin == 0 and nsoc > 1:
        ck12[nbeta:nalpha, nbeta:nalpha] = 2 * np.outer(n[nbeta:nalpha], n[nbeta:nalpha])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx, ll:ul:ndoc] = 0
        cj12[ll:ul:ndoc, ldx] = 0

        cj12[ll:ul:ndoc, ll:ul:ndoc] = 0

        ck12[ldx, ll:ul:ndoc] = np.sqrt(n_strong * n_weak)
        ck12[ll:ul:ndoc, ldx] = np.sqrt(n_strong * n_weak)

        ck12[ll:ul:ndoc, ll:ul:ndoc] = -np.sqrt(np.outer(n_weak, n_weak))

    return cj12, ck12


@njit(parallel=True, cache=True)
def der_CJCKD5(n, ista, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo):

    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5, nbf5, nv))
    Dck12r = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        Dcj12r[:, :, k] = 2 * np.outer(dn_dgamma[:, k], n)
        Dck12r[:, :, k] = np.outer(dn_dgamma[:, k], n)

    # Intrapair Electron Correlation

    for l in prange(ndoc):
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        Dcj12r[ldx, ll:ul:ndoc, :nv] = 0
        Dcj12r[ll:ul:ndoc, ldx, :nv] = 0

        Dcj12r[ll:ul:ndoc, ll:ul:ndoc, :nv] = 0

        a = max(n_strong, 10**-15)
        b = n_weak.copy()
        b[b < 10**-15] = 10**-15

        for k in range(nv):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul:ndoc, k]
            Dck12r[ldx, ll:ul:ndoc, k] = 1 / 2 * 1 / np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc, ldx, k] = 1 / 2 * 1 / np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc, ll:ul:ndoc, k] = -1 / 2 * np.outer(1 / np.sqrt(b) * dn_weak, np.sqrt(n_weak))

    return Dcj12r, Dck12r


# CJCKD7
@njit(parallel=True, cache=True)
def CJCKD7(n, ista, no1, ndoc, nsoc, nbeta, nalpha, ndns, ncwo, MSpin):
    """PNOF7 coefficients C^J and C^K that multiply J and K integrals"""

    if ista == 0:
        fi = n * (1 - n)
        fi[fi <= 0] = 0
        fi = np.sqrt(fi)
    else:
        fi = 2 * n * (1 - n)

    # Interpair Electron correlation #

    cj12 = 2 * np.outer(n, n)
    ck12 = np.outer(n, n) + np.outer(fi, fi)

    # Intrapair Electron Correlation

    if MSpin == 0 and nsoc > 1:
        ck12[nbeta:nalpha, nbeta:nalpha] = 2 * np.outer(n[nbeta:nalpha], n[nbeta:nalpha])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx, ll:ul:ndoc] = 0
        cj12[ll:ul:ndoc, ldx] = 0

        cj12[ll:ul:ndoc, ll:ul:ndoc] = 0

        ck12[ldx, ll:ul:ndoc] = np.sqrt(n_strong * n_weak)
        ck12[ll:ul:ndoc, ldx] = np.sqrt(n_strong * n_weak)

        ck12[ll:ul:ndoc, ll:ul:ndoc] = -np.sqrt(np.outer(n_weak, n_weak))

    return cj12, ck12


@njit(parallel=True, cache=True)
def der_CJCKD7(n, ista, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo):

    if ista == 0:
        fi = n * (1 - n)
        fi[fi <= 0] = 0
        fi = np.sqrt(fi)
    else:
        fi = 2 * n * (1 - n)

    dfi_dgamma = np.zeros((nbf5, nv))
    for i in prange(no1, nbf5):
        a = max(fi[i], 10**-15)
        for k in range(nv):
            if ista == 0:
                dfi_dgamma[i, k] = 1 / (2 * a) * (1 - 2 * n[i]) * dn_dgamma[i][k]
            else:
                dfi_dgamma[i, k] = 2 * (1 - 2 * n[i]) * dn_dgamma[i][k]

    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5, nbf5, nv))
    Dck12r = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        Dcj12r[:, :, k] = 2 * np.outer(dn_dgamma[:, k], n)
        Dck12r[:, :, k] = np.outer(dn_dgamma[:, k], n) + np.outer(dfi_dgamma[:, k], fi)

    # Intrapair Electron Correlation

    for l in prange(ndoc):
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        Dcj12r[ldx, ll:ul:ndoc, :nv] = 0
        Dcj12r[ll:ul:ndoc, ldx, :nv] = 0

        Dcj12r[ll:ul:ndoc, ll:ul:ndoc, :nv] = 0

        a = max(n_strong, 10**-15)
        b = n_weak.copy()
        b[b < 10**-15] = 10**-15

        for k in range(nv):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul:ndoc, k]
            Dck12r[ldx, ll:ul:ndoc, k] = 1 / 2 * 1 / np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc, ldx, k] = 1 / 2 * 1 / np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc, ll:ul:ndoc, k] = -1 / 2 * np.outer(1 / np.sqrt(b) * dn_weak, np.sqrt(n_weak))

    return Dcj12r, Dck12r


# CJCKD8
@njit(parallel=True, cache=True)
def CJCKD8(n, no1, ndoc, nsoc, nbeta, nalpha, ndns, ncwo, MSpin):
    """GNOF coefficients C^J and C^K that multiply J and K integrals"""

    h_cut = 0.02 * np.sqrt(2.0)
    n_d = np.zeros((len(n)))

    for i in prange(ndoc):
        idx = no1 + i
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[idx]
        n_weak = n[ll:ul:ndoc]

        h = 1.0 - n_strong
        coc = h / h_cut
        arg = -(coc**2)
        F = np.exp(arg)  # ! Hd/Hole
        n_d[idx] = n_strong * F
        n_d[ll:ul:ndoc] = n_weak * F  # ROd = RO*Hd/Hole

    n_d12 = np.sqrt(n_d)
    fi = n * (1 - n)
    fi[fi <= 0] = 0
    fi = np.sqrt(fi)

    # Interpair Electron correlation #

    cj12 = 2 * np.outer(n, n)
    ck12 = np.outer(n, n)

    ck12[no1:nbeta, nalpha:] += np.outer(fi[no1:nbeta], fi[nalpha:])
    ck12[nalpha:, no1:nbeta] += np.outer(fi[nalpha:], fi[no1:nbeta])
    ck12[nalpha:, nalpha:] += np.outer(fi[nalpha:], fi[nalpha:])

    # Intrapair Electron Correlation

    if MSpin == 0 and nsoc > 0:
        half = np.zeros((nsoc)) + 0.5
        ck12[no1:nbeta, nbeta:nalpha] += 0.5 * np.outer(fi[no1:nbeta], half)
        ck12[nbeta:nalpha, no1:nbeta] += 0.5 * np.outer(half, fi[no1:nbeta])
        ck12[nbeta:nalpha, nalpha:] += np.outer(half, fi[nalpha:])
        ck12[nalpha:, nbeta:nalpha] += np.outer(fi[nalpha:], half)

    if MSpin == 0 and nsoc > 1:  # then
        ck12[nbeta:nalpha, nbeta:nalpha] = 0.5

    ck12[no1:nbeta, nalpha:] += np.outer(n_d12[no1:nbeta], n_d12[nalpha:]) - np.outer(n_d[no1:nbeta], n_d[nalpha:])
    ck12[nalpha:, no1:nbeta] += np.outer(n_d12[nalpha:], n_d12[no1:nbeta]) - np.outer(n_d[nalpha:], n_d[no1:nbeta])
    ck12[nalpha:, nalpha:] += -np.outer(n_d12[nalpha:], n_d12[nalpha:]) - np.outer(n_d[nalpha:], n_d[nalpha:])

    for l in prange(ndoc):
        ldx = no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        cj12[ldx, ll:ul:ndoc] = 0
        cj12[ll:ul:ndoc, ldx] = 0

        cj12[ll:ul:ndoc, ll:ul:ndoc] = 0

        ck12[ldx, ll:ul:ndoc] = np.sqrt(n_strong * n_weak)
        ck12[ll:ul:ndoc, ldx] = np.sqrt(n_strong * n_weak)

        ck12[ll:ul:ndoc, ll:ul:ndoc] = -np.sqrt(np.outer(n_weak, n_weak))

    return cj12, ck12


@njit(parallel=True, cache=True)
def der_CJCKD8(n, dn_dgamma, no1, ndoc, nalpha, nbeta, nv, nbf5, ndns, ncwo, MSpin, nsoc):

    h_cut = 0.02 * np.sqrt(2.0)
    n_d = np.zeros((len(n)))
    dn_d_dgamma = np.zeros((len(n), nv))
    dn_d12_dgamma = np.zeros((len(n), nv))

    for i in prange(ndoc):
        idx = no1 + i
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[idx]
        n_weak = n[ll:ul:ndoc]
        dn_strong = dn_dgamma[idx, :]
        dn_weak = dn_dgamma[ll:ul:ndoc, :]

        h_idx = 1.0 - n[idx]
        coc = h_idx / h_cut
        arg = -(coc**2)
        F_idx = np.exp(arg)  # Hd/Hole
        n_d[idx] = n_strong * F_idx
        n_d[ll:ul:ndoc] = n_weak * F_idx  # n_d = RO*Hd/Hole
        dn_d_dgamma[idx, :] = F_idx * dn_strong * (1 - n_strong * (-2 * coc / h_cut))
        dn_d_dgamma[ll:ul:ndoc, :] = F_idx * (dn_weak - (-2 * coc / h_cut) * np.outer(n_weak, dn_strong))

    n_d12 = np.sqrt(n_d)
    for i in prange(nbf5):
        dn_d12_dgamma[i, :] = 0.5 * dn_d_dgamma[i, :] / max(n_d12[i], 1e-15)

    fi = n * (1 - n)
    fi[fi <= 0] = 0
    fi = np.sqrt(fi)

    dfi_dgamma = np.zeros((nbf5, nv))
    for i in prange(no1, nbf5):
        a = max(fi[i], 10**-15)
        for k in range(nv):
            dfi_dgamma[i, k] = 1 / (2 * a) * (1 - 2 * n[i]) * dn_dgamma[i][k]

    # Interpair Electron correlation #

    Dcj12r = np.zeros((nbf5, nbf5, nv))
    Dck12r = np.zeros((nbf5, nbf5, nv))
    for k in prange(nv):
        Dcj12r[:, :, k] = 2 * np.outer(dn_dgamma[:, k], n)
        Dck12r[:, :, k] = np.outer(dn_dgamma[:, k], n)

    for i in prange(nalpha, nbf5):
        Dck12r[no1:nbeta, i, :] += dfi_dgamma[no1:nbeta, :] * fi[i]
        Dck12r[i, no1:nbeta, :] += np.outer(fi[no1:nbeta], dfi_dgamma[i, :])
        Dck12r[i, nalpha:nbf5, :] += np.outer(fi[nalpha:nbf5], dfi_dgamma[i, :])

    if MSpin == 0 and nsoc > 0:
        for i in prange(nbeta, nalpha):
            Dck12r[no1:nbeta, i, :] += 0.5 * dfi_dgamma[no1:nbeta, :] * 0.5
            Dck12r[nalpha:nbf5, i, :] += dfi_dgamma[nalpha:nbf5, :] * 0.5

    if MSpin == 0 and nsoc > 1:
        Dck12r[nbeta:nalpha, nbeta:nalpha, :] = 0.0

    for i in prange(nalpha, nbf5):
        Dck12r[no1:nbeta, i, :] += dn_d12_dgamma[no1:nbeta, :] * n_d12[i] - dn_d_dgamma[no1:nbeta, :] * n_d[i]
        Dck12r[i, no1:nbeta, :] += np.outer(n_d12[no1:nbeta], dn_d12_dgamma[i, :]) - np.outer(
            n_d[no1:nbeta], dn_d_dgamma[i, :]
        )
        Dck12r[i, nalpha:nbf5, :] += -np.outer(n_d12[nalpha:nbf5], dn_d12_dgamma[i, :]) - np.outer(
            n_d[nalpha:nbf5], dn_d_dgamma[i, :]
        )

    # Intrapair Electron Correlation

    for l in prange(ndoc):
        ldx = no1 + l

        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = no1 + ndns + (ndoc - l - 1)
        ul = ll + ndoc * ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:ndoc]

        Dcj12r[ldx, ll:ul:ndoc, :nv] = 0
        Dcj12r[ll:ul:ndoc, ldx, :nv] = 0

        Dcj12r[ll:ul:ndoc, ll:ul:ndoc, :nv] = 0

        a = max(n_strong, 10**-15)
        b = n_weak.copy()
        b[b < 10**-15] = 10**-15

        for k in range(nv):
            dn_strong = dn_dgamma[ldx, k]
            dn_weak = dn_dgamma[ll:ul:ndoc, k]
            Dck12r[ldx, ll:ul:ndoc, k] = 1 / 2 * 1 / np.sqrt(a) * dn_strong * np.sqrt(n_weak)
            Dck12r[ll:ul:ndoc, ldx, k] = 1 / 2 * 1 / np.sqrt(b) * dn_weak * np.sqrt(n_strong)
            Dck12r[ll:ul:ndoc, ll:ul:ndoc, k] = -1 / 2 * np.outer(1 / np.sqrt(b) * dn_weak, np.sqrt(n_weak))

    return Dcj12r, Dck12r


def compute_2RDM(pp, n):

    norb = len(n)
    # PNOF5
    # Interpair Electron correlation #
    Id = np.identity(norb)

    inter = np.outer(n, n)
    intra = 0 * np.outer(n, n)

    # Intrapair Electron Correlation
    for l in prange(pp.ndoc):
        ldx = pp.no1 + l
        # inicio y fin de los orbitales acoplados a los fuertemente ocupados
        ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
        ul = ll + pp.ndoc * pp.ncwo

        n_strong = n[ldx]
        n_weak = n[ll:ul:pp.ndoc]

        inter[ldx, ldx] = 0
        inter[ldx, ll:ul:pp.ndoc] = 0
        inter[ll:ul:pp.ndoc, ldx] = 0
        inter[ll:ul:pp.ndoc, ll:ul:pp.ndoc] = 0

        intra[ldx, ldx] = np.sqrt(n_strong * n_strong)
        intra[ldx, ll:ul:pp.ndoc] = -np.sqrt(n_strong * n_weak)
        intra[ll:ul:pp.ndoc, ldx] = -np.sqrt(n_strong * n_weak)
        intra[ll:ul:pp.ndoc, ll:ul:pp.ndoc] = np.sqrt(np.outer(n_weak, n_weak))

    for i in range(pp.nbeta, pp.nalpha):
        inter[i, i] = 0

    Daa = np.einsum("pq,pr,qt->pqrt", inter, Id, Id, optimize=True) - np.einsum(
        "pq,pt,qr->pqrt", inter, Id, Id, optimize=True
    )
    Dab = np.einsum("pq,pr,qt->pqrt", inter, Id, Id, optimize=True) + np.einsum(
        "pr,pq,rt->pqrt", intra, Id, Id, optimize=True
    )

    # PNOF7
    if pp.ipnof == 7 or pp.ipnof == 8:
        fi = n * (1 - n)
        fi[fi <= 0] = 0
        fi = np.sqrt(fi)
        Pi_s = np.outer(fi, fi)
        # Intrapair Electron Correlation
        for l in prange(pp.ndoc):
            ldx = pp.no1 + l
            # inicio y fin de los orbitales acoplados a los fuertemente ocupados
            ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
            ul = ll + pp.ndoc * pp.ncwo

            Pi_s[ldx, ldx] = 0
            Pi_s[ldx, ll:ul:pp.ndoc] = 0
            Pi_s[ll:ul:pp.ndoc, ldx] = 0
            Pi_s[ll:ul:pp.ndoc, ll:ul:pp.ndoc] = 0
        for i in range(pp.nbeta, pp.nalpha):
            Pi_s[i, i] = 0

        inter2 = 0 * Pi_s.copy()
        inter2[pp.nbeta : pp.nalpha, pp.nbeta : pp.nalpha] = Pi_s[pp.nbeta : pp.nalpha, pp.nbeta : pp.nalpha]
        Pi_s[pp.nbeta : pp.nalpha, pp.nbeta : pp.nalpha] = 0

        Dab -= np.einsum("pq,pt,qr->pqrt", inter2, Id, Id, optimize=True)

        if pp.ipnof == 8:
            Pi_s[: pp.nbeta, : pp.nbeta] = 0
            Pi_s[: pp.nbeta, pp.nbeta : pp.nalpha] *= 0.5
            Pi_s[pp.nbeta : pp.nalpha, : pp.nbeta] *= 0.5

        Dab -= np.einsum("pr,pq,rt->pqrt", Pi_s, Id, Id, optimize=True)

        if pp.ipnof == 8:

            h_cut = 0.02 * np.sqrt(2.0)
            n_d = np.zeros((len(n)))

            for i in prange(pp.ndoc):
                idx = pp.no1 + i
                # inicio y fin de los orbitales acoplados a los fuertemente ocupados
                ll = pp.no1 + pp.ndns + (pp.ndoc - i - 1)
                ul = ll + pp.ndoc * pp.ncwo

                n_strong = n[idx]
                n_weak = n[ll:ul:pp.ndoc]

                h = 1.0 - n_strong
                coc = h / h_cut
                arg = -(coc**2)
                F = np.exp(arg)  # ! Hd/Hole
                n_d[idx] = n_strong * F
                n_d[ll:ul:pp.ndoc] = n_weak * F  # ROd = RO*Hd/Hole

            n_d12 = np.sqrt(n_d)

            inter = np.outer(n_d12, n_d12) - np.outer(n_d, n_d)
            inter2 = np.outer(n_d12, n_d12) + np.outer(n_d, n_d)
            # Intrapair Electron Correlation
            for l in prange(pp.ndoc):
                ldx = pp.no1 + l
                # inicio y fin de los orbitales acoplados a los fuertemente ocupados
                ll = pp.no1 + pp.ndns + (pp.ndoc - l - 1)
                ul = ll + pp.ndoc * pp.ncwo

                inter[ldx, ldx] = 0
                inter[ldx, ll:ul:pp.ndoc] = 0
                inter[ll:ul:pp.ndoc, ldx] = 0
                inter[ll:ul:pp.ndoc, ll:ul:pp.ndoc] = 0
                inter2[ldx, ldx] = 0
                inter2[ldx, ll:ul:pp.ndoc] = 0
                inter2[ll:ul:pp.ndoc, ldx] = 0
                inter2[ll:ul:pp.ndoc, ll:ul:pp.ndoc] = 0

            inter[pp.nbeta :, pp.nbeta :] = 0
            inter[: pp.nalpha, : pp.nalpha] = 0
            inter2[: pp.nalpha, :] = 0
            inter2[:, : pp.nalpha] = 0

            Pi_d = inter - inter2

            Dab -= np.einsum("pr,pq,rt->pqrt", Pi_d, Id, Id, optimize=True)

    return Daa, Dab


# Creamos un seleccionador de PNOF


def PNOFi_selector(n, p):
    if p.ipnof == 4:
        cj12, ck12 = CJCKD4(n, p.no1, p.ndoc, p.nsoc, p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
    if p.ipnof == 5:
        cj12, ck12 = CJCKD5(n, p.no1, p.ndoc, p.nsoc, p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
    if p.ipnof == 7:
        cj12, ck12 = CJCKD7(n, p.ista, p.no1, p.ndoc, p.nsoc, p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)
    if p.ipnof == 8:
        cj12, ck12 = CJCKD8(n, p.no1, p.ndoc, p.nsoc, p.nbeta, p.nalpha, p.ndns, p.ncwo, p.MSpin)

    return cj12, ck12


def der_PNOFi_selector(n, dn_dgamma, p):
    if p.ipnof == 4:
        Dcj12r, Dck12r = der_CJCKD4(
            n, p.ista, dn_dgamma, p.no1, p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo
        )
    if p.ipnof == 5:
        Dcj12r, Dck12r = der_CJCKD5(
            n, p.ista, dn_dgamma, p.no1, p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo
        )
    if p.ipnof == 7:
        Dcj12r, Dck12r = der_CJCKD7(
            n, p.ista, dn_dgamma, p.no1, p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo
        )
    if p.ipnof == 8:
        Dcj12r, Dck12r = der_CJCKD8(
            n, dn_dgamma, p.no1, p.ndoc, p.nalpha, p.nbeta, p.nv, p.nbf5, p.ndns, p.ncwo, p.MSpin, p.nsoc
        )

    return Dcj12r, Dck12r


# @njit(cache=True)
def ocupacion(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin, occ_method):
    """Code to get occupation numbers from parameters according to the choosen encoding"""

    if occ_method == "Trigonometric":
        n, dn_dgamma = ocupacion_trigonometric(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin)
    elif occ_method == "Softmax":
        n, dn_dgamma = ocupacion_softmax(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin)
    elif occ_method == "EBI":
        n, dn_dgamma = ocupacion_ebi(gamma, ndoc)
    return n, dn_dgamma


@njit(cache=True)
def ocupacion_trigonometric(gamma, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
    """Transform gammas to n according to the trigonometric
    parameterization of the occupation numbers"""

    n = np.zeros((nbf5))
    dn_dgamma = np.zeros((nbf5, nv))
    dni_dgammai = np.zeros((nbf5))

    n[0:no1] = 1  # [1,no1]

    n[no1 : no1 + ndoc] = 1 / 2 * (1 + np.cos(gamma[:ndoc]) ** 2)  # (no1,no1+ndoc]
    dni_dgammai[no1 : no1 + ndoc] = -1 / 2 * np.sin(2 * gamma[:ndoc])
    for i in range(ndoc):
        # dn_g/dgamma_g
        dn_dgamma[no1 + i, i] = dni_dgammai[no1 + i]

    if not HighSpin:
        n[no1 + ndoc : no1 + ndns] = 0.5  # (no1+ndoc,no1+ndns]
    elif HighSpin:
        n[no1 + ndoc : no1 + ndns] = 1.0  # (no1+ndoc,no1+ndns]

    h = 1 - n
    for i in range(ndoc):
        ll_n = no1 + ndns + (ndoc - i - 1)
        ul_n = ll_n + ndoc * ncwo
        n_pi = n[ll_n:ul_n:ndoc]
        ll_gamma = ndoc + (ndoc - i - 1)
        ul_gamma = ll_gamma + ndoc * (ncwo - 1)
        gamma_pi = gamma[ll_gamma:ul_gamma:ndoc]

        # n_pi
        n_pi[:] = h[no1 + i]
        for kw in range(ncwo - 1):
            n_pi[kw] *= np.sin(gamma_pi[kw]) ** 2
            n_pi[kw + 1 :] *= np.cos(gamma_pi[kw]) ** 2

        # dn_pi/dgamma_g
        dn_pi_dgamma_g = dn_dgamma[ll_n:ul_n:ndoc, i]
        dn_pi_dgamma_g[:] = -dni_dgammai[no1 + i]
        for kw in range(ncwo - 1):
            dn_pi_dgamma_g[kw] *= np.sin(gamma_pi[kw]) ** 2
            dn_pi_dgamma_g[kw + 1 :] *= np.cos(gamma_pi[kw]) ** 2

        # dn_pi/dgamma_pj (j<i)
        dn_pi_dgamma_pj = dn_dgamma[ll_n:ul_n:ndoc, ll_gamma:ul_gamma:ndoc]
        for jw in range(ncwo - 1):
            dn_pi_dgamma_pj[jw + 1 :, jw] = n[no1 + i] - 1
            for kw in range(jw):
                dn_pi_dgamma_pj[jw + 1 :, jw] *= np.cos(gamma_pi[kw]) ** 2
            dn_pi_dgamma_pj[jw + 1 :, jw] *= np.sin(2 * gamma_pi[jw])
            for kw in range(jw + 1, ncwo - 1):
                dn_pi_dgamma_pj[kw, jw] *= np.sin(gamma_pi[kw]) ** 2
                dn_pi_dgamma_pj[kw + 1 :, jw] *= np.cos(gamma_pi[kw]) ** 2

        # dn_pi/dgamma_i
        for jw in range(ncwo - 1):
            dn_pi_dgamma_pj[jw, jw] = 1 - n[no1 + i]
        for kw in range(ncwo - 1):
            dn_pi_dgamma_pj[kw, kw] *= np.sin(2 * gamma_pi[kw])
            for lw in range(kw + 1, ncwo - 1):
                dn_pi_dgamma_pj[lw, lw] *= np.cos(gamma_pi[kw]) ** 2

    return n, dn_dgamma


@njit(cache=True)
def ocupacion_softmax(x, no1, ndoc, nalpha, nv, nbf5, ndns, ncwo, HighSpin):
    """Transform gammas to n according to the softmax
    parameterization of the occupation numbers"""

    n = np.zeros((nbf5))
    dn_dx = np.zeros((nbf5, nv))

    if not HighSpin:
        n[no1 + ndoc : no1 + ndns] = 0.5  # (no1+ndoc,no1+ndns]
    elif HighSpin:
        n[no1 + ndoc : no1 + ndns] = 1.0  # (no1+ndoc,no1+ndns]

    exp_x = np.exp(x)

    for i in range(ndoc):
        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ndoc * ncwo
        n_pi = n[ll:ul:ndoc]

        ll_x = (ndoc -i - 1)
        ul_x = ll_x + ndoc * ncwo
        dn_g_dx_pi = dn_dx[i, ll_x:ul_x:ndoc]
        dn_pi_dx_pi = dn_dx[ll:ul:ndoc, ll_x:ul_x:ndoc]
        dn_pi_dx_g = dn_dx[ll:ul:ndoc, i]

        exp_x_pi = exp_x[ll_x:ul_x:ndoc]

        den = 1 + np.sum(exp_x_pi)

        n[i] = 1 / den
        n_pi[:] = exp_x_pi / den

        dn_g_dx_pi[:] = -n[i] * n_pi
        dn_pi_dx_pi[:, :] = -np.outer(n_pi, n_pi)

        for j in range(ncwo):
            dn_pi_dx_pi[j, j] += n_pi[j]

    return n, dn_dx


# @jit(cache=True)
def D_deviation_ebi(mu, x, N):
    """Derivative from gamma of the loss function used in ebi

    df/dmu = 2 * (N- sum_p n_p) * (-sum_p dn_p/dmu)

    """

    nv = np.shape(x)[0]

    # n_p
    sum_n = 0
    for i in range(nv):
        sum_n += (1 + math.erf(x[i] + mu[0])) / 2

    # dn_p/dmu = 1/sqrt(pi) * exp(-(x_p + mu)^2)
    sum_dn = 0
    for i in range(nv):
        sum_dn += 1 / np.sqrt(np.pi) * np.exp(-((x[i] + mu[0]) ** 2))

    D_dev = -2 * (N - sum_n) * sum_dn

    return D_dev


# @jit(cache=True)
def deviation_ebi(mu, x, N):
    """Loss function used in ebi: Squared deviation from N/2 of the
    occupation numbers

    f = (N - \sum_i n_i)^2

    """

    nv = np.shape(x)[0]

    n = np.zeros(nv)
    for i in range(nv):
        n[i] += (1 + math.erf(x[i] + mu[0])) / 2
    sum_n = np.sum(n)

    dev = (N - sum_n) ** 2

    return dev


# @jit(cache=True)
def ocupacion_ebi(x, N):
    """Transform gammas to n according to the ebi
    parameterization of the occupation numbers

    n_p = (1 + erf(x_p + mu)) / 2

    dn_p/dx_p = erf'(x_p + \mu) * (1 + d\mu/dx_p) / 2
    dn_p/dx_q = erf'(x_p + \mu) * d\mu/dx_q / 2

    dmu/dx_p = - erf'(x_p + \mu)/(sum_q erf'(x_q + \mu))
    """

    nv = np.shape(x)[0]

    mu = [0.0]
    res = minimize(deviation_ebi, mu, args=(x, N), jac=D_deviation_ebi, method="CG")
    mu = res.x[0]
    dev = res.fun
    nit = res.nit
    success = res.success

    # n_p
    n = np.zeros(nv)
    for i, xi in enumerate(x):
        n[i] = (1 + math.erf(xi + mu)) / 2

    # dmu/dx_p
    den = 0
    for i in range(nv):
        den += np.exp(-((x[i] + mu) ** 2))
    D_mu_x = np.zeros(nv)
    for i in range(nv):
        D_mu_x[i] = -np.exp(-((x[i] + mu) ** 2)) / den

    # dn_p/dx_
    D_n_x = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(nv):
            if i == j:
                D_n_x[i, j] = 1 / np.sqrt(np.pi) * np.exp(-((x[i] + mu) ** 2)) * (1 + D_mu_x[j])
            else:
                D_n_x[i, j] = 1 / np.sqrt(np.pi) * np.exp(-((x[i] + mu) ** 2)) * D_mu_x[j]

    return n, D_n_x


def calce(n, cj12, ck12, J_MO, K_MO, H_core, p):

    E = 0

    if p.MSpin == 0:

        # 2H + J
        E = E + 2 * np.einsum("i,i", n, H_core, optimize=True)
        E = E + np.einsum("i,i", n[: p.nbeta], np.diagonal(J_MO)[: p.nbeta], optimize=True)
        E = E + np.einsum("i,i", n[p.nalpha : p.nbf5], np.diagonal(J_MO)[p.nalpha : p.nbf5], optimize=True)

        # C^J JMO
        np.fill_diagonal(cj12, 0)  # Remove diag.
        E = E + np.einsum("ij,ji->", cj12, J_MO, optimize=True)  # sum_ij

        # C^K KMO
        np.fill_diagonal(ck12, 0)  # Remove diag.
        E = E - np.einsum("ij,ji->", ck12, K_MO, optimize=True)  # sum_ij

    elif not p.MSpin == 0:
        E = 0

        # 2H + J
        E = E + np.einsum(
            "i,i", n[: p.nbeta], 2 * H_core[: p.nbeta] + np.diagonal(J_MO)[: p.nbeta], optimize=True
        )  # [0,Nbeta]
        E = E + np.einsum("i,i", n[p.nbeta : p.nalpha], 2 * H_core[p.nbeta : p.nalpha], optimize=True)  # (Nbeta,Nalpha]
        E = E + np.einsum(
            "i,i",
            n[p.nalpha : p.nbf5],
            2 * H_core[p.nalpha : p.nbf5] + np.diagonal(J_MO)[p.nalpha : p.nbf5],
            optimize=True,
        )  # (Nalpha,Nbf5)

        # C^J JMO
        np.fill_diagonal(cj12, 0)  # Remove diag.
        E = E + np.einsum("ij,ji->", cj12[: p.nbeta, : p.nbeta], J_MO[: p.nbeta, : p.nbeta], optimize=True)  # sum_ij
        E = E + np.einsum(
            "ij,ji->", cj12[: p.nbeta, p.nalpha : p.nbf5], J_MO[p.nalpha : p.nbf5, : p.nbeta], optimize=True
        )  # sum_ij
        E = E + np.einsum(
            "ij,ji->", cj12[p.nalpha : p.nbf5, : p.nbeta], J_MO[: p.nbeta, p.nalpha : p.nbf5], optimize=True
        )  # sum_ij
        E = E + np.einsum(
            "ij,ji->",
            cj12[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            J_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            optimize=True,
        )  # sum_ij

        # C^K KMO
        np.fill_diagonal(ck12, 0)  # Remove diag.
        E = E - np.einsum("ij,ji->", ck12[: p.nbeta, : p.nbeta], K_MO[: p.nbeta, : p.nbeta], optimize=True)  # sum_ij
        E = E - np.einsum(
            "ij,ji->", ck12[: p.nbeta, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, : p.nbeta], optimize=True
        )  # sum_ij
        E = E - np.einsum(
            "ij,ji->", ck12[p.nalpha : p.nbf5, : p.nbeta], K_MO[: p.nbeta, p.nalpha : p.nbf5], optimize=True
        )  # sum_ij
        E = E - np.einsum(
            "ij,ji->",
            ck12[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            optimize=True,
        )  # sum_ij

        # n JMO
        E = E + 2 * np.einsum("i,ji->", n[: p.nbeta], J_MO[p.nbeta : p.nalpha, : p.nbeta], optimize=True)  # sum_ij
        E = E + 2 * np.einsum(
            "i,ji->", n[p.nalpha : p.nbf5], J_MO[p.nbeta : p.nalpha, p.nalpha : p.nbf5], optimize=True
        )  # sum_ij
        E = E + 0.5 * (
            np.einsum("i,ji->", n[p.nbeta : p.nalpha], J_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha], optimize=True)
            - np.einsum("i,ii->", n[p.nbeta : p.nalpha], J_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha], optimize=True)
        )

        # n KMO
        E = E - np.einsum("i,ji->", n[: p.nbeta], K_MO[p.nbeta : p.nalpha, : p.nbeta], optimize=True)  # sum_ij
        E = E - np.einsum(
            "i,ji->", n[p.nalpha : p.nbf5], K_MO[p.nbeta : p.nalpha, p.nalpha : p.nbf5], optimize=True
        )  # sum_ij
        E = E - 0.5 * (
            np.einsum("i,ji->", n[p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha], optimize=True)
            + np.einsum("i,ii->", n[p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha], optimize=True)
        )

    return E


def calcocce(gamma, J_MO, K_MO, H_core, p):

    n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    cj12, ck12 = PNOFi_selector(n, p)

    E = calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

    return E


def calcoccg(gamma, J_MO, K_MO, H_core, p):

    grad = np.zeros((p.nv))

    n, dn_dgamma = ocupacion(gamma, p.no1, p.ndoc, p.nalpha, p.nv, p.nbf5, p.ndns, p.ncwo, p.HighSpin, p.occ_method)
    Dcj12r, Dck12r = der_PNOFi_selector(n, dn_dgamma, p)

    if p.MSpin == 0:

        # dn_dgamma (2H+J)
        grad += np.einsum("ik,i->k", dn_dgamma, 2 * H_core + np.diagonal(J_MO), optimize=True)

        # 2 dCJ_dgamma J_MO
        diag = np.diag_indices(p.nbf5)
        Dcj12r[diag] = 0
        grad += 2 * np.einsum("ijk,ji->k", Dcj12r, J_MO, optimize=True)

        # -2 dCK_dgamma K_MO
        diag = np.diag_indices(p.nbf5)
        Dck12r[diag] = 0
        grad -= 2 * np.einsum("ijk,ji->k", Dck12r, K_MO, optimize=True)

    elif not p.MSpin == 0:

        # dn_dgamma (2H+J)
        grad += np.einsum(
            "ik,i->k",
            dn_dgamma[p.no1 : p.nbeta, : p.nv],
            2 * H_core[p.no1 : p.nbeta] + np.diagonal(J_MO)[p.no1 : p.nbeta],
            optimize=True,
        )  # [0,Nbeta]
        grad += np.einsum(
            "ik,i->k",
            dn_dgamma[p.nalpha : p.nbf5, : p.nv],
            2 * H_core[p.nalpha : p.nbf5] + np.diagonal(J_MO)[p.nalpha : p.nbf5],
            optimize=True,
        )  # [Nalpha,Nbf5]

        # 2 dCJ_dgamma J_MO
        Dcj12r[np.diag_indices(p.nbf5)] = 0
        grad += 2 * np.einsum(
            "ijk,ji->k", Dcj12r[p.no1 : p.nbeta, : p.nbeta, : p.nv], J_MO[: p.nbeta, p.no1 : p.nbeta], optimize=True
        )
        grad += 2 * np.einsum(
            "ijk,ji->k",
            Dcj12r[p.no1 : p.nbeta, p.nalpha : p.nbf5, : p.nv],
            J_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta],
            optimize=True,
        )
        grad += 2 * np.einsum(
            "ijk,ji->k", Dcj12r[p.nalpha : p.nbf5, : p.nbeta, : p.nv], J_MO[: p.nbeta, p.nalpha : p.nbf5], optimize=True
        )
        grad += 2 * np.einsum(
            "ijk,ji->k",
            Dcj12r[p.nalpha : p.nbf5, p.nalpha : p.nbf5, : p.nv],
            J_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            optimize=True,
        )

        # -2 dCK_dgamma K_MO
        Dck12r[np.diag_indices(p.nbf5)] = 0
        grad -= 2 * np.einsum(
            "ijk,ji->k", Dck12r[p.no1 : p.nbeta, : p.nbeta, : p.nv], K_MO[: p.nbeta, p.no1 : p.nbeta], optimize=True
        )
        grad -= 2 * np.einsum(
            "ijk,ji->k",
            Dck12r[p.no1 : p.nbeta, p.nalpha : p.nbf5, : p.nv],
            K_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta],
            optimize=True,
        )
        grad -= 2 * np.einsum(
            "ijk,ji->k", Dck12r[p.nalpha : p.nbf5, : p.nbeta, : p.nv], K_MO[: p.nbeta, p.nalpha : p.nbf5], optimize=True
        )
        grad -= 2 * np.einsum(
            "ijk,ji->k",
            Dck12r[p.nalpha : p.nbf5, p.nalpha : p.nbf5, : p.nv],
            K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5],
            optimize=True,
        )

        # 2 dn_dgamma J_MO
        grad += 2 * np.einsum(
            "jk,ij->k", dn_dgamma[p.no1 : p.nbeta, : p.nv], J_MO[p.nbeta : p.nalpha, p.no1 : p.nbeta], optimize=True
        )
        grad += 2 * np.einsum(
            "jk,ij->k", dn_dgamma[p.nalpha : p.nbf5, : p.nv], J_MO[p.nbeta : p.nalpha, p.nalpha : p.nbf5], optimize=True
        )

        # - dn_dgamma K_MO
        grad -= np.einsum(
            "jk,ij->k", dn_dgamma[p.no1 : p.nbeta, : p.nv], K_MO[p.nbeta : p.nalpha, p.no1 : p.nbeta], optimize=True
        )
        grad -= np.einsum(
            "jk,ij->k", dn_dgamma[p.nalpha : p.nbf5, : p.nv], K_MO[p.nbeta : p.nalpha, p.nalpha : p.nbf5], optimize=True
        )

    return grad


def calcorbe(y, n, cj12, ck12, C, H, I, b_mnl, p):

    Cnew = pynof.rotate_orbital(y, C, p)

    J_MO, K_MO, H_core = pynof.computeJKH_MO(Cnew, H, I, b_mnl, p)

    E = calce(n, cj12, ck12, J_MO, K_MO, H_core, p)

    return E


def calcorbg(y, n, cj12, ck12, C, H, I, b_mnl, p):

    Cnew = pynof.rotate_orbital(y, C, p)

    elag, Hmat = pynof.computeLagrange2(n, cj12, ck12, Cnew, H, I, b_mnl, p)

    grad = 4 * (elag - elag.T)

    grads = np.zeros((p.nvar))

    k = 0
    for i in range(p.nbf5):
        for j in range(i + 1, p.nbf):
            grads[k] = grad[i, j]
            k += 1
    grad = grads

    return grad


def calcorbeg(y, n, cj12, ck12, C, H, I, b_mnl, p):

    Cnew = pynof.rotate_orbital(y, C, p)

    elag, Hmat = pynof.computeLagrange2(n, cj12, ck12, Cnew, H, I, b_mnl, p)

    grad = 4 * (elag - elag.T)

    grads = np.zeros((int(p.nbf * (p.nbf - 1) / 2) - int(p.no0 * (p.no0 - 1) / 2)))

    k = 0
    for i in range(p.nbf5):
        for j in range(i + 1, p.nbf):
            grads[k] = grad[i, j]
            k += 1
    grad = grads

    E = np.einsum("ii", elag[: p.nbf5, : p.nbf5], optimize=True)
    E = E + np.einsum("i,ii", n[: p.nbf5], Hmat[: p.nbf5, : p.nbf5], optimize=True)

    return E, grad
