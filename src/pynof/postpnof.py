import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.optimize import root
from time import time
import pynof
from numba import prange, njit, jit
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, minres
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh
from scipy.linalg import eig
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.special import roots_legendre


def build_XmY_XpY(L, tempm, bigomega, nab):

    XmY = np.dot(L, tempm)

    XpY = np.linalg.solve(np.transpose(L), tempm)

    XmY = np.einsum("mi,i->mi", XmY, 1 / np.sqrt(bigomega), optimize=True)
    XpY = np.einsum("mi,i->mi", XpY, np.sqrt(bigomega), optimize=True)

    return XmY, XpY


@njit(parallel=True, cache=True)
def get_AmB(eig, nalpha, nbf):
    nvir = nbf - nalpha
    AmB = np.zeros((nalpha, nvir, nalpha, nvir))
    for i in prange(nalpha):
        for aidx, a in enumerate(range(nalpha, nbf)):
            AmB[i, aidx, i, aidx] = eig[a] - eig[i]

    return AmB


def build_AmB_ApB(eig, pqrt_at, nalpha, nbf, nab):

    nvir = nbf - nalpha

    ApB = 4 * pqrt_at[0:nalpha, nalpha:nbf, 0:nalpha, nalpha:nbf]
    AmB = get_AmB(eig, nalpha, nbf)

    ApB = ApB + AmB

    EcRPA = -0.25 * (np.einsum("iaia->", ApB, optimize=True) + np.einsum("iaia->", AmB, optimize=True))

    ApB = ApB.reshape(nab, nab)
    AmB = AmB.reshape(nab, nab)

    return EcRPA, AmB, ApB


@njit(parallel=True, cache=True)
def ccsd_eq(eig, pqrt, pqrt_at, nbeta, nalpha, nbf):
    # CCSD
    Eccsd = 0
    FockM_in = np.zeros((nbf, nbf))
    for pp in range(nbf):
        FockM_in[pp, pp] = eig[pp]
    ERItmp = np.zeros((nbf, nbf, nbf, nbf))

    nocc = max(nbeta, nalpha)
    nocc2 = 2 * nocc
    nbf2 = 2 * nbf
    nvir2 = nbf2 - nocc2

    FockM = np.zeros((nbf2, nbf2))
    for i in range(nbf2):
        for j in range(nbf2):
            if i % 2 == j % 2:
                FockM[j, i] = FockM_in[slbasis(j), slbasis(i)]

    td = np.zeros((nocc2, nocc2, nvir2, nvir2))
    for i in range(nocc2):
        for j in range(nocc2):
            for aidx, a in enumerate(range(nocc2, nbf2)):
                for bidx, b in enumerate(range(nocc2, nbf2)):
                    td[i, j, aidx, bidx] = spin_int(i, j, a, b, pqrt_at) / (
                        FockM[i, i] + FockM[j, j] - FockM[a, a] - FockM[b, b] + 1e-8
                    )

    deltaE = 1
    itera = 0
    ts = np.zeros((nocc2, nvir2))
    print("")
    while np.abs(deltaE) > 1e-6 and itera < 1000000:

        Fmi, Wmnij, Fae, Wabef, Fme, Wmbej = ccsd_update_interm(ts, td, FockM, pqrt_at, nocc2, nvir2, nbf2)

        tsnew, tdnew = ccsd_update_t1_t2(ts, td, Fmi, Wmnij, Fae, Wabef, Fme, Wmbej, FockM, pqrt_at, nocc2, nvir2, nbf2)

        for i in range(nocc2):
            for j in range(nocc2):
                for aidx, a in enumerate(range(nocc2, nbf2)):
                    if j == 0:
                        if np.abs(tsnew[i, aidx]) < 1e-8:
                            tsnew[i, aidx] = 0
                    for bidx, b in enumerate(range(nocc2, nbf2)):
                        if np.abs(tdnew[i, j, aidx, bidx]) < 1e-8:
                            tdnew[i, j, aidx, bidx] = 0.0
        ts = tsnew.copy()
        td = tdnew.copy()

        ccsd_en_nof = 0
        nsocc = 2 * nalpha
        ndocc = 2 * nbeta
        for aidx, a in enumerate(range(nsocc, nbf2)):
            for bidx, b in enumerate(range(nsocc, nbf2)):
                for i in range(ndocc):
                    if b == nsocc:
                        ccsd_en_nof += FockM[i, a] * ts[i, aidx]
                    for j in range(ndocc):
                        ccsd_en_nof += (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                    for j in range(ndocc, nsocc):
                        ccsd_en_nof += 0.5 * (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                for i in range(ndocc, nsocc):
                    for j in range(ndocc):
                        ccsd_en_nof += 0.5 * (
                            0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                            + 0.5 * spin_int(i, j, aidx, bidx, pqrt) * ts[i, aidx] * ts[j, bidx]
                        )
                    for j in range(ndocc, nsocc):
                        if i % 2 == 0:
                            k = i + 1
                        else:
                            k = i - 1
                        if j != i and j != k:
                            ccsd_en_nof += 0.25 * (
                                0.25 * spin_int(i, j, a, b, pqrt) * td[i, j, aidx, bidx]
                                + 0.5 * spin_int(i, j, a, b, pqrt) * ts[i, aidx] * ts[j, bidx]
                            )
        Eccsd_new = ccsd_en_nof
        deltaE = Eccsd - Eccsd_new
        Eccsd = Eccsd_new
        itera += 1
        if itera % 2 == 0:
            # print("CCSD corr. energy {:5.3f} iter {:d} deltaE {:5.3f}".format(Eccsd,itera,deltaE))
            print(" ....CCSD corr. energy", Eccsd, " iter", itera, "deltaE", deltaE)
        # break
    # print("CCSD procedure finished after {:5.3f} iter. with deltaE {:5.3f}".format(itera,deltaE))
    print(" ....CCSD procedure finished after", itera, "iter. with deltaE", deltaE)
    print("")
    EcCCSD = Eccsd_new

    return EcCCSD


@njit(parallel=True, cache=True)
def ccsd_update_interm(ts, td, FockM, pqrt_at, nocc2, nvir2, nbf2):
    Fmi = np.zeros((nocc2, nocc2))
    Wmnij = np.zeros((nocc2, nocc2, nocc2, nocc2))
    for m in range(nocc2):
        for i in range(nocc2):
            if i != m:
                Fmi[m, i] = FockM[m, i]
            for n in range(nocc2):
                for j in range(nocc2):
                    Wmnij[m, n, i, j] = spin_int(m, n, i, j, pqrt_at)
                    for eidx, e in enumerate(range(nocc2, nbf2)):
                        if j == 0:
                            Fmi[m, i] += ts[n, eidx] * spin_int(m, n, i, e, pqrt_at)
                            if n == 0:
                                Fmi[m, i] += 0.5 * ts[i, eidx] * FockM[m, e]
                        Wmnij[m, n, i, j] += ts[j, eidx] * spin_int(m, n, i, e, pqrt_at) - ts[i, eidx] * spin_int(
                            m, n, j, e, pqrt_at
                        )
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            tau = td[i, j, eidx, fidx] + ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                            Wmnij[m, n, i, j] += 0.25 * tau * spin_int(m, n, e, f, pqrt_at)
                            if j == 0:
                                taus = td[i, n, eidx, fidx] + 0.5 * (
                                    ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                                )
                                Fmi[m, i] += 0.5 * taus * spin_int(m, n, e, f, pqrt_at)

    Fae = np.zeros((nvir2, nvir2))
    Wabef = np.zeros((nvir2, nvir2, nvir2, nvir2))
    for aidx, a in enumerate(range(nocc2, nbf2)):
        for eidx, e in enumerate(range(nocc2, nbf2)):
            if a != e:
                Fae[aidx, eidx] = FockM[a, e]
            for bidx, b in enumerate(range(nocc2, nbf2)):
                for fidx, f in enumerate(range(nocc2, nbf2)):
                    Wabef[aidx, bidx, eidx, fidx] = spin_int(a, b, e, f, pqrt_at)
                    for m in range(nocc2):
                        if b == nocc2:
                            if f == nocc2:
                                Fae[aidx, eidx] += -0.5 * ts[m, aidx] * FockM[m, e]
                            Fae[aidx, eidx] += ts[m, fidx] * spin_int(m, a, f, e, pqrt_at)
                        Wabef[aidx, bidx, eidx, fidx] += -ts[m, bidx] * spin_int(a, m, e, f, pqrt_at) + ts[
                            m, aidx
                        ] * spin_int(b, m, e, f, pqrt_at)
                        for n in range(nocc2):
                            tau = td[m, n, aidx, bidx] + ts[m, aidx] * ts[n, bidx] - ts[m, bidx] * ts[n, aidx]
                            Wabef[aidx, bidx, eidx, fidx] += 0.25 * tau * spin_int(m, n, e, f, pqrt_at)
                            if b == nocc2:
                                taus = td[m, n, aidx, fidx] + 0.5 * (
                                    ts[m, aidx] * ts[n, fidx] - ts[m, fidx] * ts[n, aidx]
                                )
                                Fae[aidx, eidx] += -0.5 * taus * spin_int(m, n, e, f, pqrt_at)

    Fme = np.zeros((nocc2, nvir2))
    Wmbej = np.zeros((nocc2, nvir2, nvir2, nocc2))
    for m in range(nocc2):
        for eidx, e in enumerate(range(nocc2, nbf2)):
            Fme[m, eidx] = FockM[m, e]
            for bidx, b in enumerate(range(nocc2, nbf2)):
                for j in range(nocc2):
                    Fme[m, eidx] += ts[j, bidx] * spin_int(m, j, e, b, pqrt_at)
                    Wmbej[m, bidx, eidx, j] = spin_int(m, b, e, j, pqrt_at)
                    for fidx, f in enumerate(range(nocc2, nbf2)):
                        Wmbej[m, bidx, eidx, j] += ts[j, fidx] * spin_int(m, b, e, f, pqrt_at)
                    for n in range(nocc2):
                        Wmbej[m, bidx, eidx, j] += -ts[n, bidx] * spin_int(m, n, e, j, pqrt_at)
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            Wmbej[m, bidx, eidx, j] += -(
                                0.5 * td[j, n, fidx, bidx] + ts[j, fidx] * ts[n, bidx]
                            ) * spin_int(m, n, e, f, pqrt_at)

    return Fmi, Wmnij, Fae, Wabef, Fme, Wmbej


@njit(parallel=True, cache=True)
def ccsd_update_t1_t2(ts, td, Fmi, Wmnij, Fae, Wabef, Fme, Wmbej, FockM, pqrt_at, nocc2, nvir2, nbf2):
    tsnew = np.zeros((nocc2, nvir2))
    tdnew = np.zeros((nocc2, nocc2, nvir2, nvir2))

    for aidx, a in enumerate(range(nocc2, nbf2)):
        for bidx, b in enumerate(range(nocc2, nbf2)):
            for i in range(nocc2):
                # T1
                if b == nocc2:
                    tsnew[i, aidx] = FockM[i, a]
                    for m in range(nocc2):
                        tsnew[i, aidx] -= ts[m, aidx] * Fmi[m, i]
                        for eidx, e in enumerate(range(nocc2, nbf2)):
                            tsnew[i, aidx] += td[i, m, aidx, eidx] * Fme[m, eidx]
                            tsnew[i, aidx] += -ts[m, eidx] * spin_int(m, a, i, e, pqrt_at)
                            if m == 0:
                                tsnew[i, aidx] += ts[i, eidx] * Fae[aidx, eidx]
                            for fidx, f in enumerate(range(nocc2, nbf2)):
                                tsnew[i, aidx] += -0.5 * td[i, m, eidx, fidx] * spin_int(m, a, e, f, pqrt_at)
                            for n in range(nocc2):
                                tsnew[i, aidx] += -0.5 * td[m, n, aidx, eidx] * spin_int(n, m, e, i, pqrt_at)
                    #                    if(qnewton):
                    #                        ria=tsnew[i,a]-ts[i,a]*(FockM[i,i]-FockM[a,a])
                    #                        if(abs(ria)>tol8):
                    #                            tsnew[i,a] += ria/(FockM[i,i]-FockM[a,a]+1e-8)
                    #                    else:
                    tsnew[i, aidx] = tsnew[i, aidx] / (FockM[i, i] - FockM[a, a] + 1e-8)

                # T2
                for j in range(nocc2):
                    tdnew[i, j, aidx, bidx] = spin_int(i, j, a, b, pqrt_at)
                    for eidx, e in enumerate(range(nocc2, nbf2)):
                        tdnew[i, j, aidx, bidx] += (
                            td[i, j, aidx, eidx] * Fae[bidx, eidx] - td[i, j, bidx, eidx] * Fae[aidx, eidx]
                        )
                        for m in range(nocc2):
                            tdnew[i, j, aidx, bidx] += (
                                -0.5 * td[i, j, aidx, eidx] * ts[m, bidx] * Fme[m, eidx]
                                + 0.5 * td[i, j, bidx, eidx] * ts[m, aidx] * Fme[m, eidx]
                            )
                            if e == nocc2:
                                tdnew[i, j, aidx, bidx] += (
                                    -td[i, m, aidx, bidx] * Fmi[m, j] + td[j, m, aidx, bidx] * Fmi[m, i]
                                )
                                tdnew[i, j, aidx, bidx] += -ts[m, aidx] * spin_int(m, b, i, j, pqrt_at) + ts[
                                    m, bidx
                                ] * spin_int(m, a, i, j, pqrt_at)
                                for n in range(nocc2):
                                    tau = td[m, n, aidx, bidx] + ts[m, aidx] * ts[n, bidx] - ts[m, bidx] * ts[n, aidx]
                                    tdnew[i, j, aidx, bidx] += 0.5 * tau * Wmnij[m, n, i, j]
                            tdnew[i, j, aidx, bidx] += (
                                -0.5 * td[i, m, aidx, bidx] * ts[j, eidx] * Fme[m, eidx]
                                + 0.5 * td[j, m, aidx, bidx] * ts[i, eidx] * Fme[m, eidx]
                            )
                            tdnew[i, j, aidx, bidx] += td[i, m, aidx, eidx] * Wmbej[m, bidx, eidx, j] - ts[
                                i, eidx
                            ] * ts[m, aidx] * spin_int(m, b, e, j, pqrt_at)
                            tdnew[i, j, aidx, bidx] += -td[j, m, aidx, eidx] * Wmbej[m, bidx, eidx, i] + ts[
                                j, eidx
                            ] * ts[m, aidx] * spin_int(m, b, e, i, pqrt_at)
                            tdnew[i, j, aidx, bidx] += -td[i, m, bidx, eidx] * Wmbej[m, aidx, eidx, j] + ts[
                                i, eidx
                            ] * ts[m, bidx] * spin_int(m, a, e, j, pqrt_at)
                            tdnew[i, j, aidx, bidx] += td[j, m, bidx, eidx] * Wmbej[m, aidx, eidx, i] - ts[
                                j, eidx
                            ] * ts[m, bidx] * spin_int(m, a, e, i, pqrt_at)
                        tdnew[i, j, aidx, bidx] += ts[i, eidx] * spin_int(a, b, e, j, pqrt_at) - ts[j, eidx] * spin_int(
                            a, b, e, i, pqrt_at
                        )
                        for fidx, f in enumerate(range(nocc2, nbf2)):
                            tau = td[i, j, eidx, fidx] + ts[i, eidx] * ts[j, fidx] - ts[i, fidx] * ts[j, eidx]
                            tdnew[i, j, aidx, bidx] += 0.5 * tau * Wabef[aidx, bidx, eidx, fidx]
                    tdnew[i, j, aidx, bidx] = tdnew[i, j, aidx, bidx] / (
                        FockM[i, i] + FockM[j, j] - FockM[a, a] - FockM[b, b] + 1e-8
                    )

    return tsnew, tdnew


@njit(parallel=True, cache=True)
def slbasis(i):
    if i % 2 == 0:
        return int(i / 2)
    else:
        return int((i + 1) / 2) - 1


@njit(parallel=True, cache=True)
def spin_int(p, q, r, s, ERImol):
    value1, value2 = 0, 0
    if p % 2 == r % 2 and q % 2 == s % 2:
        value1 = ERImol[slbasis(s), slbasis(p), slbasis(r), slbasis(q)]
    if p % 2 == s % 2 and q % 2 == r % 2:
        value2 = ERImol[slbasis(r), slbasis(p), slbasis(s), slbasis(q)]
    return value1 - value2


@njit(parallel=True, cache=True)
def W(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, cfreq):

    w_iajb = eri_at
    for s in range(nab):
        w_iajb += wmn_at[i, a, s] * wmn_at[j, b, s] * (1 / (cfreq - bigomega[s]) - 1 / (cfreq + bigomega[s]))

    return w_iajb


@njit(parallel=True, cache=True)
def integrated_omega(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order):

    integral = 0
    for ii in prange(order):
        integral += (
            W(i, a, j, b, eri_at, wmn_at, eig, nab, bigomega, cfreqs[ii])
            * weights[ii]
            * 2
            * (eig[i] - eig[a])
            / ((eig[i] - eig[a]) ** 2 + freqs[ii] ** 2)
            * 2
            * (eig[j] - eig[b])
            / ((eig[j] - eig[b]) ** 2 + freqs[ii] ** 2)
        )

    if np.abs(integral.imag) > 1e-4:
        print("Warning, large imaginary component", integral)

    integral = integral.real

    return integral


@njit(parallel=True, cache=True)
def rpa_sosex(freqs, weights, sumw, order, wmn_at, eig, pqrt, pqrt_at, bigomega, nab, nbeta, nalpha, nbf):

    iEcRPA = 0
    iEcSOSEX = 0

    weights = weights / sumw
    freqs = (freqs + 1) / 2

    weights = weights / (1 - freqs) ** 2
    freqs = freqs / (1 - freqs)
    cfreqs = freqs * 1j

    for a in prange(nalpha, nbf):
        for b in prange(nalpha, nbf):
            for i in prange(nbeta):
                for j in prange(nbeta):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -pqrt[i, a, j, b] * integral
                    iEcSOSEX += pqrt[i, b, j, a] * integral
                for j in prange(nbeta, nalpha):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.5 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.5 * pqrt[i, b, j, a] * integral
            for i in prange(nbeta, nalpha):
                for j in prange(nbeta):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.5 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.5 * pqrt[i, b, j, a] * integral
                for j in prange(nbeta, nalpha):
                    integral = integrated_omega(
                        i, a, j, b, pqrt_at[i, a, j, b], wmn_at, eig, nab, bigomega, weights, freqs, cfreqs, order
                    )
                    iEcRPA += -0.25 * pqrt[i, a, j, b] * integral
                    iEcSOSEX += 0.25 * pqrt[i, b, j, a] * integral

    iEcRPA = iEcRPA / np.pi
    iEcSOSEX = 0.5 * iEcSOSEX / np.pi

    return iEcRPA, iEcSOSEX


@njit(parallel=True, cache=True)
def gw_gm_eq(wmn_at, pqrt, eig, bigomega, XpY, nab, nbeta, nalpha, nbf):
    EcGoWo = 0
    EcGMSOS = 0

    for a in prange(nalpha, nbf):
        for b in range(nalpha, nbf):
            for i in range(nbeta):
                for j in range(nbeta):
                    l = j * (nbf - nalpha) + (b - nalpha)
                    for s in range(nab):
                        EcGoWo += (
                            wmn_at[i, a, s]
                            * pqrt[i, a, j, b]
                            * XpY[l, s]
                            * np.sqrt(2)
                            / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                        )
                        EcGMSOS -= (
                            wmn_at[i, a, s]
                            * pqrt[i, b, j, a]
                            * XpY[l, s]
                            * np.sqrt(2)
                            / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                        )
    if nalpha != nbeta:
        for a in range(nalpha, nbf):
            for b in range(nalpha, nbf):
                for i in range(nbeta):
                    for j in range(nbeta, nalpha):
                        l = j * (nbf - nalpha) + (b - nalpha)
                        for s in range(nab):
                            EcGoWo += (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, a, j, b]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                            EcGMSOS -= (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, b, j, a]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                for i in range(nbeta, nalpha):
                    for j in range(nbeta, nalpha):
                        l = j * (nbf - nalpha) + (b - nalpha)
                        for s in range(nab):
                            EcGoWo += (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, a, j, b]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                            EcGMSOS -= (
                                0.5
                                * wmn_at[i, a, s]
                                * pqrt[i, b, j, a]
                                * XpY[l, s]
                                * np.sqrt(2)
                                / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                            )
                    for j in range(nbeta, nalpha):
                        if i != j:
                            l = j * (nbf - nalpha) + (b - nalpha)
                            for s in range(nab):
                                EcGoWo += (
                                    0.25
                                    * wmn_at[i, a, s]
                                    * pqrt[i, a, j, b]
                                    * XpY[l, s]
                                    * np.sqrt(2)
                                    / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                                )
                                EcGMSOS -= (
                                    0.25
                                    * wmn_at[i, a, s]
                                    * pqrt[i, b, j, a]
                                    * XpY[l, s]
                                    * np.sqrt(2)
                                    / (eig[i] - eig[a] - bigomega[s] + 1e-10)
                                )

    return EcGoWo, EcGMSOS


def build_wmn(pqrt, pqrt_at, XpY, nab, nalpha, nbf):

    nvir = nbf - nalpha
    XpY = XpY.reshape(nalpha, nvir, nab)
    wmn = np.einsum("pqia,iak->pqk", pqrt[:, :, :nalpha, nalpha:], XpY, optimize=True)
    wmn_at = np.einsum("pqia,iak->pqk", pqrt_at[:, :, :nalpha, nalpha:], XpY, optimize=True)

    return wmn, wmn_at


def td_polarizability(EcRPA, C, Dipole, bigomega, XpY, nab, nalpha, nbf):

    Dipole_MO = np.einsum("mi,qmn,nj->qij", C, Dipole, C, optimize=True)

    nvir = nbf - nalpha
    XpY = XpY.reshape(nalpha, nvir, nab)

    tempm = np.sqrt(2) * np.einsum("qia,iak->qk", Dipole_MO[:, :nalpha, nalpha:nbf], XpY, optimize=True)
    oscstr = 2 / 3 * np.einsum("qk,qk,k->k", tempm, tempm, bigomega, optimize=True)
    EcRPA += 0.5 * np.sum(np.abs(bigomega))

    print("TD-H (RPA) CASIDA eq. solved")
    print("N. excitation   a.u.         eV            nm      osc. strenght")
    for i in range(nab):
        if oscstr[i] > 10**-6:
            print(
                "{:^ 10d} {: 11.5f} {: 11.5f} {: 11.4f} {: 11.6f} ".format(
                    i, bigomega[i], bigomega[i] * 27.211399, 1239.84193 / (bigomega[i] * 27.211399), oscstr[i]
                )
            )
    print("")

    print("Static Polarizability:")
    print("")
    staticpol = 2 * np.einsum("ji,ki,i->jk", tempm, tempm, 1 / (bigomega + 1e-10), optimize=True)
    for xyz in range(3):
        print("{:15.5f} {:15.5f} {:15.5f}".format(staticpol[xyz, 0], staticpol[xyz, 1], staticpol[xyz, 2]))
    print("")
    print(
        "Trace of the static polarizability {:9.5f}".format((staticpol[0, 0] + staticpol[1, 1] + staticpol[2, 2]) / 3)
    )
    print("")

    return EcRPA


def build_F_MO(C, H, I, b_mnl, p):
    D = pynof.computeD_HF(C, p)
    if p.MSpin == 0:
        if p.nsoc > 0:
            Dalpha = pynof.computeDalpha_HF(C, p)
            D = D + 0.5 * Dalpha
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = H + 2 * J - K
        EHFL = np.trace(np.matmul(D, H) + np.matmul(D, F))
    elif not p.MSpin == 0:
        Dalpha = pynof.computeDalpha_HF(C, p)
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = 2 * J - K
        EHFL = 2 * np.trace(np.matmul(D + 0.5 * Dalpha, H)) + np.trace(np.matmul(D + Dalpha, F))
        F = H + F
        if p.nsoc > 1:
            J, K = pynof.computeJK_HF(0.5 * Dalpha, I, b_mnl, p)
            Falpha = J - K
            EHFL = EHFL + 2 * np.trace(np.matmul(0.5 * Dalpha, Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(C), F), C)

    # eig = np.einsum("ii->i",F_MO) #JFHLY: Print info

    return EHFL, F_MO


@njit(parallel=True, cache=True)
def ERIS_attenuated(pqrt, Cintra, Cinter, no1, ndoc, nsoc, ndns, ncwo, nbf5, nbf):
    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1 + i] = no1 + i
        ll = no1 + ndns + ncwo * (ndoc - i - 1)
        ul = no1 + ndns + ncwo * (ndoc - i)
        subspaces[ll:ul] = no1 + i
    for i in range(nsoc):
        subspaces[no1 + ndoc + i] = no1 + ndoc + i
    subspaces[nbf5:] = -1

    pqrt_at = np.zeros((nbf, nbf, nbf, nbf))
    for p in prange(nbf):
        for q in prange(nbf):
            for r in prange(nbf):
                for t in prange(nbf):
                    if (
                        subspaces[p] == subspaces[q]
                        and subspaces[p] == subspaces[r]
                        and subspaces[p] == subspaces[t]
                        and subspaces[p] != -1
                    ):
                        pqrt_at[p, q, r, t] = pqrt[p, q, r, t] * Cintra[p] * Cintra[q] * Cintra[r] * Cintra[t]
                    else:
                        pqrt_at[p, q, r, t] = pqrt[p, q, r, t] * Cinter[p] * Cinter[q] * Cinter[r] * Cinter[t]

    return pqrt_at


@njit(parallel=True, cache=True)
def F_MO_attenuated(F_MO, Cintra, Cinter, no1, nalpha, ndoc, nsoc, ndns, ncwo, nbf5, nbf):

    F_MO_at = np.zeros((nbf, nbf))

    subspaces = np.zeros((nbf))
    for i in range(no1):
        subspaces[i] = i
    for i in range(ndoc):
        subspaces[no1 + i] = no1 + i
        ll = no1 + ndns + ncwo * (ndoc - i - 1)
        ul = no1 + ndns + ncwo * (ndoc - i)
        subspaces[ll:ul] = no1 + i
    for i in range(nsoc):
        subspaces[no1 + ndoc + i] = no1 + ndoc + i
    subspaces[nbf5:] = -1

    for p in prange(nbf):
        for q in prange(nbf):
            if p != q:
                if subspaces[p] == subspaces[q]:
                    F_MO_at[p, q] = F_MO[p, q] * Cintra[p] * Cintra[q]
                else:
                    F_MO_at[p, q] = F_MO[p, q] * Cinter[p] * Cinter[q]
            else:
                F_MO_at[p, q] = F_MO[p, q]
    F_MO_at[:nalpha, nalpha:nbf] = 0.0
    F_MO_at[nalpha:nbf, :nalpha] = 0.0

    return F_MO_at


@njit(parallel=True, cache=True)
def mp2_eq(eig, pqrt, pqrt_at, nbeta, nalpha, nbf):

    EcMP2 = 0
    for a in prange(nalpha, nbf):
        for b in prange(nalpha, nbf):
            for i in prange(nbeta):
                for j in prange(nbeta):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (2 * pqrt_at[i, a, j, b] - pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
                for j in prange(nbeta, nalpha):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
            for i in prange(nbeta, nalpha):
                for j in prange(nbeta):
                    EcMP2 += (
                        pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )
                for j in prange(nbeta, nalpha):
                    EcMP2 += (
                        0.5
                        * pqrt[i, a, j, b]
                        * (pqrt_at[i, a, j, b] - 0.5 * pqrt_at[i, b, j, a])
                        / (eig[i] + eig[j] - eig[a] - eig[b] + 1e-10)
                    )

    return EcMP2


def ECorrNonDyn(n, C, H, I, b_mnl, p):
    fi = 2 * n * (1 - n)

    CK12nd = np.outer(fi, fi)

    beta = np.sqrt((1 - abs(1 - 2 * n)) * n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo * (p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo * (p.ndoc - l)
        CK12nd[p.no1 + l, ll:ul] = beta[p.no1 + l] * beta[ll:ul]
        CK12nd[ll:ul, p.no1 + l] = beta[ll:ul] * beta[p.no1 + l]
        CK12nd[ll:ul, ll:ul] = -np.outer(beta[ll:ul], beta[ll:ul])

    # C^K KMO
    J_MO, K_MO, H_core = pynof.computeJKH_MO(C, H, I, b_mnl, p)

    ECndHF = 0
    ECndl = 0
    if p.MSpin == 0:
        ECndHF = -np.einsum(
            "ii,ii->", CK12nd[p.nbeta : p.nalpha, p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha]
        )  # sum_ij
        ECndl -= np.einsum("ij,ji->", CK12nd, K_MO)  # sum_ij
        ECndl += np.einsum("ii,ii->", CK12nd, K_MO)  # Quita i=j
    elif not p.MSpin == 0:
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # Quita i=j
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # Quita i=j

    return ECndHF, ECndl


def compare(original, new, byelement=False):

    if byelement:
        original_lin = original.flatten()
        new_lin = new.flatten()

        for ol, nl in zip(original_lin, new_lin):
            if np.abs(ol - nl) > 1e-5:
                print(ol, nl, ol - nl)

    print("Original pqrt", np.sum(original))
    print("New pqrt", np.sum(new))
    print("Diff pqrt", np.sum(new - original))
    print("Abs Diff pqrt", np.sum(np.abs(new - original)))


def mbpt(n, C, H, I, b_mnl, Dipole, E_nuc, E_elec, p):

    #####################################################
    #
    # Reference Article
    # Coupling Natural Orbital Functional Theory
    # and Many-Bodyi Perturbation Theory by Using
    # Nondynamically Correlated Canonical Orbitals
    # https://doi.org/10.1021/acs.jctc.1c00858
    #
    #####################################################

    t1 = time()

    print(" MBPT")
    print("=========")

    nab = p.nvir * p.nalpha
    last_coup = p.nalpha + p.ncwo * (p.ndoc)

    print("Number of orbitals        (NBASIS) = {}".format(p.nbf))
    print("Number of frozen pairs       (NFR) = {}".format(p.no1))
    print("Number of occupied orbs.     (NOC) = {}".format(p.nalpha))
    print("Number of virtual orbs.     (NVIR) = {}".format(p.nvir))
    print("Size of A+B and A-B (NAB=NOCxNVIR) = {}".format(nab))
    print("")

    occ = np.zeros((p.nbf))
    occ[: p.nbf5] = n

    print(" ....Building F_MO")
    EHFL, F_MO = build_F_MO(C, H, I, b_mnl, p)

    print(" ....Transforming ERIs mnsl->pqrt")
    pqrt = pynof.compute_pqrt(C, I, b_mnl, p)

    # Eq. (17) and (18)
    Cintra = 1 - (1 - abs(1 - 2 * occ)) ** 2
    Cinter = abs(1 - 2 * occ) ** 2
    Cinter[: p.nalpha] = 1.0

    # Eq. (19) Canonicalization of NOs Step 1
    print(" ....Attenuating F_MO")
    F_MO_at = F_MO_attenuated(F_MO, Cintra, Cinter, p.no1, p.nalpha, p.ndoc, p.nsoc, p.ndns, p.ncwo, p.nbf5, p.nbf)

    # Eq. (20)
    print(" ....Attenuating pqrt")
    pqrt_at = ERIS_attenuated(pqrt, Cintra, Cinter, p.no1, p.ndoc, p.nsoc, p.ndns, p.ncwo, p.nbf5, p.nbf)

    # Canonicalization of NOs Step 2 and 3
    eig, C_can = eigh(F_MO_at)

    # Canonicalization of NOs step 4
    print(" ....Canonicalizing pqrt")
    pqrt = np.einsum("pqrt,pm,qn,rs,tl->mnsl", pqrt, C_can, C_can, C_can, C_can, optimize=True)
    print(" ....Canonicalizing pqrt_at")
    pqrt_at = np.einsum("pqrt,pm,qn,rs,tl->mnsl", pqrt_at, C_can, C_can, C_can, C_can, optimize=True)
    print("")

    print("List of qp-orbital energies (a.u.) and occ numbers used")
    print()
    mu = (eig[p.nalpha] + eig[p.nalpha - 1]) / 2
    eig = eig - mu
    for i in range(p.nbf):
        print(" {: 8.6f} {:5.3f}".format(eig[i], occ[i]))
    print("Chemical potential used for qp-orbital energies: {}".format(mu))
    print("")

    # Supp. Info. Eq. (4) and (5)
    EcRPA, AmB, ApB = build_AmB_ApB(eig, pqrt_at, p.nalpha, p.nbf, nab)

    L = sp.linalg.cholesky(ApB, lower=True)

    bigomega2, tempm = np.linalg.eigh(np.matmul(np.matmul(np.transpose(L), AmB), L))
    #    bigomega2,tempm = np.linalg.eigh(np.matmul(ApB,AmB))

    bigomega = np.sqrt(bigomega2)

    XmY, XpY = build_XmY_XpY(L, tempm, bigomega, nab)

    # Eq (8) second term
    print(" ....Computing Polarizabilities")
    EcRPA = td_polarizability(EcRPA, C, Dipole, bigomega, XpY, nab, p.nalpha, p.nbf)

    # Supp. Info. Eq. (6)
    print(" ....Computing wmn")
    wmn, wmn_at = build_wmn(pqrt, pqrt_at, XpY, nab, p.nalpha, p.nbf)

    # Eq. (27) and (29)
    print(" ....Computing gw_gm")
    EcGoWo, EcGMSOS = gw_gm_eq(wmn_at, pqrt, eig, bigomega, XpY, nab, p.nbeta, p.nalpha, p.nbf)

    EcGoWo *= 2
    EcGoWoSOS = EcGoWo + EcGMSOS

    # Eq. (22)
    print(" ....Computing mp2")
    EcMP2 = mp2_eq(eig, pqrt, pqrt_at, p.nbeta, p.nalpha, p.nbf)

    # Eq. (24) and (28)
    order = 40
    freqs, weights, sumw = roots_legendre(order, mu=True)
    print(" ....Computing rpa_sosex")
    iEcRPA, iEcSOSEX = (
        0,
        0,
    )  # rpa_sosex(freqs,weights,sumw,order,wmn_at,eig,pqrt,pqrt_at,bigomega,nab,p.nbeta,p.nalpha,p.nbf)

    iEcRPASOS = iEcRPA + iEcSOSEX

    # Eq. (26)
    print(" ....Computing ccsd")
    pqrt_at = np.einsum("pqsr->sqpr", pqrt_at, optimize=True)
    pqrt = np.einsum("pqsr->sqpr", pqrt, optimize=True)
    EcCCSD = 0  # ccsd_eq(eig,pqrt,pqrt_at,p.nbeta,p.nalpha,p.nbf)

    ECndHF, ECndl = ECorrNonDyn(n, C, H, I, b_mnl, p)

    ESD = EHFL + E_nuc + ECndHF
    ESDc = ESD + ECndl
    EPNOF = E_elec + E_nuc

    print(" E(SD)                = {: f}".format(ESD))
    print(" E(SD+ND)             = {: f}".format(ESDc))
    print(" E(PNOFi)             = {: f}".format(EPNOF))
    print("")
    print(" Ec(ND)               = {: f}".format(ECndl))
    print(" Ec(RPA-FURCHE)       = {: f}".format(EcRPA))
    print(" Ec(RPA)              = {: f}".format(iEcRPA))
    print(" Ec(AC-SOSEX)         = {: f}".format(iEcSOSEX))
    print(" Ec(RPA+AC-SOSEX)     = {: f}".format(iEcRPASOS))
    print(" Ec(GW@GM)            = {: f}".format(EcGoWo))
    print(" Ec(SOSEX@GM)         = {: f}".format(EcGMSOS))
    print(" Ec(GW@GM+SOSEX@GM)   = {: f}".format(EcGoWoSOS))
    print(" Ec(MP2)              = {: f}".format(EcMP2))
    print(" Ec(CCSD)             = {: f}".format(EcCCSD))
    print("")
    print(" E(RPA-FURCHE)       = {: f}".format(ESD + EcRPA))
    print(" E(RPA)              = {: f}".format(ESD + iEcRPA))
    print(" E(RPA+AC-SOSEX)     = {: f}".format(ESD + iEcRPASOS))
    print(" E(GW@GM)            = {: f}".format(ESD + EcGoWo))
    print(" E(SOSEX@GM)         = {: f}".format(ESD + EcGMSOS))
    print(" E(GW@GM+SOSEX@GM)   = {: f}".format(ESD + EcGoWoSOS))
    print(" E(MP2)              = {: f}".format(ESD + EcMP2))
    print(" E(CCSD)             = {: f}".format(ESD + EcCCSD))
    print("")
    print(" E(NOF-c-RPA-FURCHE)       = {: f}".format(ESDc + EcRPA))
    print(" E(NOF-c-RPA)              = {: f}".format(ESDc + iEcRPA))
    print(" E(NOF-c-RPA+AC+SOSEX)     = {: f}".format(ESDc + iEcRPASOS))
    print(" E(NOF-c-GW@GM)            = {: f}".format(ESDc + EcGoWo))
    print(" E(NOF-c-SOSEX@GM)         = {: f}".format(ESDc + EcGMSOS))
    print(" E(NOF-c-GW@GM+SOSEX@GM)   = {: f}".format(ESDc + EcGoWoSOS))
    print(" E(NOF-c-MP2)              = {: f}".format(ESDc + EcMP2))
    print(" E(NOF-c-CCSD)             = {: f}".format(ESDc + EcCCSD))

    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2 - t1))


def nofmp2(n, C, H, I, b_mnl, E_nuc, p):

    t1 = time()

    print(" NOF-MP2")
    print("=========")

    occ = n[p.no1 : p.nbf5]
    vec = C[:, p.no1 : p.nbf]

    D = pynof.computeD_HF(C, p)
    if p.MSpin == 0:
        if p.nsoc > 0:
            Dalpha = pynof.computeDalpha_HF(C, p)
            D = D + 0.5 * Dalpha
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = H + 2 * J - K
        EHFL = np.trace(np.matmul(D, H) + np.matmul(D, F))
    elif not p.MSpin == 0:
        Dalpha = pynof.computeDalpha_HF(C, p)
        J, K = pynof.computeJK_HF(D, I, b_mnl, p)
        F = 2 * J - K
        EHFL = 2 * np.trace(np.matmul(D + 0.5 * Dalpha, H)) + np.trace(np.matmul(D + Dalpha, F))
        F = H + F
        if p.nsoc > 1:
            J, K = pynof.computeJK_HF(0.5 * Dalpha, I, b_mnl, p)
            Falpha = J - K
            EHFL = EHFL + 2 * np.trace(np.matmul(0.5 * Dalpha, Falpha))
            F = F + Falpha

    F_MO = np.matmul(np.matmul(np.transpose(vec), F), vec)

    eig = np.einsum("ii->i", F_MO[: p.nbf - p.no1, : p.nbf - p.no1])

    iajb = pynof.compute_iajb(C, I, b_mnl, p)
    FI1 = np.ones(p.nbf - p.no1)
    FI2 = np.ones(p.nbf - p.no1)

    FI1[: p.nbf5 - p.no1] = 1 - (1 - abs(1 - 2 * occ[: p.nbf5 - p.no1])) ** 2

    FI2[p.nalpha - p.no1 : p.nbf5 - p.no1] = abs(1 - 2 * occ[p.nalpha - p.no1 : p.nbf5 - p.no1]) ** 2

    Tijab = CalTijab(iajb, F_MO, eig, FI1, FI2, p)
    ECd = 0
    for k in range(p.nvir):
        for l in range(p.nvir):
            for i in range(p.ndoc):
                for j in range(p.ndoc):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (2 * Tijab[ijkl] - Tijab[ijlk])
                for j in range(p.ndoc, p.ndns):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk])
            for i in range(p.ndoc, p.ndns):
                for j in range(p.ndoc):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk])
                for j in range(p.ndoc, p.ndns):
                    Xijkl = iajb[j, k, i, l]
                    ijkl = i + j * p.ndns + k * p.ndns * p.ndns + l * p.ndns * p.ndns * p.nvir
                    ijlk = i + j * p.ndns + l * p.ndns * p.ndns + k * p.ndns * p.ndns * p.nvir
                    if j != i:
                        ECd = ECd + Xijkl * (Tijab[ijkl] - 0.5 * Tijab[ijlk]) / 2

    fi = 2 * n * (1 - n)

    CK12nd = np.outer(fi, fi)

    beta = np.sqrt((1 - abs(1 - 2 * n)) * n)

    for l in range(p.ndoc):
        ll = p.no1 + p.ndns + p.ncwo * (p.ndoc - l - 1)
        ul = p.no1 + p.ndns + p.ncwo * (p.ndoc - l)
        CK12nd[p.no1 + l, ll:ul] = beta[p.no1 + l] * beta[ll:ul]
        CK12nd[ll:ul, p.no1 + l] = beta[ll:ul] * beta[p.no1 + l]
        CK12nd[ll:ul, ll:ul] = -np.outer(beta[ll:ul], beta[ll:ul])

    # C^K KMO
    J_MO, K_MO, H_core = pynof.computeJKH_MO(C, H, I, b_mnl, p)

    ECndHF = 0
    ECndl = 0
    if p.MSpin == 0:
        ECndHF = -np.einsum(
            "ii,ii->", CK12nd[p.nbeta : p.nalpha, p.nbeta : p.nalpha], K_MO[p.nbeta : p.nalpha, p.nbeta : p.nalpha]
        )  # sum_ij
        ECndl -= np.einsum("ij,ji->", CK12nd, K_MO)  # sum_ij
        ECndl += np.einsum("ii,ii->", CK12nd, K_MO)  # Quita i=j
    elif not p.MSpin == 0:
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.no1 : p.nbeta, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.no1 : p.nbeta]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl -= np.einsum(
            "ij,ji->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # sum_ij
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.no1 : p.nbeta, p.no1 : p.nbeta], K_MO[p.no1 : p.nbeta, p.no1 : p.nbeta]
        )  # Quita i=j
        ECndl += np.einsum(
            "ii,ii->", CK12nd[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5]
        )  # Quita i=j

    print("      Ehfc      = {:f}".format(EHFL + E_nuc + ECndHF))
    print("")
    print("      ECd       = {:f}".format(ECd))
    print("      ECnd      = {:f}".format(ECndl))
    print("      Ecorre    = {:f}".format(ECd + ECndl))
    print("      E(NOFMP2) = {:f}".format(EHFL + ECd + ECndl + E_nuc + ECndHF))
    print("")

    t2 = time()
    print("Elapsed Time: {:10.2f} (Seconds)".format(t2 - t1))


def CalTijab(iajb, F_MO, eig, FI1, FI2, p):

    print("Starting CalTijab")

    B = build_B(iajb, FI1, FI2, p.ndoc, p.ndns, p.nvir, p.ncwo)
    print("....B vector Computed")

    Tijab = Tijab_guess(iajb, eig, p.ndoc, p.ndns, p.nvir)
    print("....Tijab Guess Computed")

    # A_CSR = csr_matrix(build_A(F_MO,FI1,FI2,p.no1,p.ndoc,p.ndns,p.nvir,p.ncwo,p.nbf))
    # print("A matrix has {}/{} elements with Tol = {}".format(len(A),p.nvir**4*p.ndoc**4,1e-10))
    # Tijab = solve_Tijab(A_CSR,B,Tijab,p)

    res = root(build_R, Tijab, args=(B, F_MO, FI1, FI2, p.no1, p.ndoc, p.ndns, p.nvir, p.ncwo, p.nbf), method="krylov")
    if res.success:
        print("....Tijab found as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
    else:
        print("....WARNING! Tijab NOT FOUND as a Root of R = B - A*Tijab in {} iterations".format(res.nit))
        print(res)
    Tijab = res.x
    print("")

    return Tijab


@njit(parallel=True, cache=True)
def build_A(F_MO, FI1, FI2, no1, ndoc, ndns, nvir, ncwo, nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo * (ndoc - i - 1)
        ul = ncwo * (ndoc - i)
        npair[ll:ul] = i + 1

    A = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)))
    IROW = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)), dtype=np.int32)
    ICOL = np.empty((2 * ndns**2 * nvir**2 * (nbf - no1)), dtype=np.int32)

    nnz = -1
    for ib in range(nvir):
        for ia in range(nvir):
            for j in range(ndns):
                for i in range(ndns):
                    # print(nnz)
                    jab = (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    iab = i + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    ijb = i + (j) * ndns + (ib) * ndns * ndns * nvir
                    ija = i + (j) * ndns + (ia) * ndns * ndns
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir

                    nnz = nnz + 1
                    A[nnz] = F_MO[ia + ndns, ia + ndns] + F_MO[ib + ndns, ib + ndns] - F_MO[i, i] - F_MO[j, j]
                    IROW[nnz] = ijab
                    ICOL[nnz] = i + jab

                    for k in range(i):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            nnz += 1
                            A[nnz] = -Cki * F_MO[i, k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k + jab
                            nnz += 1
                            A[nnz] = -Cki * F_MO[i, k]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k + jab

                    for k in range(j):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            nnz += 1
                            A[nnz] = -Ckj * F_MO[j, k]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns + iab
                            nnz += 1
                            A[nnz] = -Ckj * F_MO[j, k]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns + iab

                    for k in range(ia):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            nnz += 1
                            A[nnz] = Ckia * F_MO[ia + ndns, k + ndns]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns * ndns + ijb
                            nnz += 1
                            A[nnz] = Ckia * F_MO[ia + ndns, k + ndns]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns * ndns + ijb

                    for k in range(ib):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            nnz += 1
                            A[nnz] = Ckib * F_MO[ib + ndns, k + ndns]
                            IROW[nnz] = ijab
                            ICOL[nnz] = k * ndns * ndns * nvir + ija
                            nnz += 1
                            A[nnz] = Ckib * F_MO[ib + ndns, k + ndns]
                            ICOL[nnz] = ijab
                            IROW[nnz] = k * ndns * ndns * nvir + ija

    A = A[: nnz + 1]
    IROW = IROW[: nnz + 1]
    ICOL = ICOL[: nnz + 1]

    return A, (IROW, ICOL)


@njit(parallel=True, cache=True)
def build_R(T, B, F_MO, FI1, FI2, no1, ndoc, ndns, nvir, ncwo, nbf):
    npair = np.zeros((nvir))
    for i in range(ndoc):
        ll = ncwo * (ndoc - i - 1)
        ul = ncwo * (ndoc - i)
        npair[ll:ul] = i + 1

    Bp = np.zeros((ndns**2 * nvir**2))

    for ib in prange(nvir):
        for ia in prange(nvir):
            for j in prange(ndns):
                for i in prange(ndns):
                    jab = (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    iab = i + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    ijb = i + (j) * ndns + (ib) * ndns * ndns * nvir
                    ija = i + (j) * ndns + (ia) * ndns * ndns
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir

                    Bp[ijab] += (F_MO[ia + ndns, ia + ndns] + F_MO[ib + ndns, ib + ndns] - F_MO[i, i] - F_MO[j, j]) * T[
                        i + jab
                    ]

                    for k in range(i):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            Bp[ijab] += (-Cki * F_MO[i, k]) * T[k + jab]
                    for k in range(i + 1, ndns):
                        if abs(F_MO[i, k]) > 1e-10:
                            Cki = FI2[k] * FI2[i]
                            Bp[ijab] += (-Cki * F_MO[i, k]) * T[k + jab]

                    for k in range(j):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            Bp[ijab] += (-Ckj * F_MO[j, k]) * T[k * ndns + iab]
                    for k in range(j + 1, ndns):
                        if abs(F_MO[j, k]) > 1e-10:
                            Ckj = FI2[k] * FI2[j]
                            Bp[ijab] += (-Ckj * F_MO[j, k]) * T[k * ndns + iab]

                    for k in range(ia):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            Bp[ijab] += (Ckia * F_MO[ia + ndns, k + ndns]) * T[k * ndns * ndns + ijb]
                    for k in range(ia + 1, nvir):
                        if abs(F_MO[ia + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ia]:
                                Ckia = FI1[k + ndns] * FI1[ia + ndns]
                            else:
                                Ckia = FI2[k + ndns] * FI2[ia + ndns]
                            Bp[ijab] += (Ckia * F_MO[ia + ndns, k + ndns]) * T[k * ndns * ndns + ijb]

                    for k in range(ib):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            Bp[ijab] += (Ckib * F_MO[ib + ndns, k + ndns]) * T[k * ndns * ndns * nvir + ija]
                    for k in range(ib + 1, nvir):
                        if abs(F_MO[ib + ndns, k + ndns]) > 1e-10:
                            if npair[k] == npair[ib]:
                                Ckib = FI1[k + ndns] * FI1[ib + ndns]
                            else:
                                Ckib = FI2[k + ndns] * FI2[ib + ndns]
                            Bp[ijab] += (Ckib * F_MO[ib + ndns, k + ndns]) * T[k * ndns * ndns * nvir + ija]

    R = B - Bp
    return R


@njit(parallel=True, cache=True)
def build_B(iajb, FI1, FI2, ndoc, ndns, nvir, ncwo):
    B = np.zeros((ndns**2 * nvir**2))
    for i in prange(ndns):
        lmin_i = ndns + ncwo * (ndns - i - 1)
        lmax_i = ndns + ncwo * (ndns - i - 1) + ncwo
        for j in range(ndns):
            if i == j:
                for k in range(nvir):
                    ik = i + k * ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        if lmin_i <= kn and kn < lmax_i and lmin_i <= ln and ln < lmax_i:
                            Ciikl = FI1[kn] * FI1[ln] * FI1[i] * FI1[i]
                        else:
                            Ciikl = FI2[kn] * FI2[ln] * FI2[i] * FI2[i]
                        iikl = i + i * ndns + k * ndns * ndns + l * ndns * ndns * nvir
                        B[iikl] = -Ciikl * iajb[i, k, i, l]
            else:
                for k in range(nvir):
                    ik = i + k * ndns
                    kn = k + ndns
                    for l in range(nvir):
                        ln = l + ndns
                        ijkl = i + j * ndns + k * ndns * ndns + l * ndns * ndns * nvir
                        Cijkl = FI2[kn] * FI2[ln] * FI2[i] * FI2[j]
                        B[ijkl] = -Cijkl * iajb[j, k, i, l]
    return B


@njit(parallel=True, cache=True)
def Tijab_guess(iajb, eig, ndoc, ndns, nvir):
    Tijab = np.zeros(nvir**2 * ndns**2)
    for ia in prange(nvir):
        for i in range(ndns):
            for ib in range(nvir):
                for j in range(ndns):
                    ijab = i + (j) * ndns + (ia) * ndns * ndns + (ib) * ndns * ndns * nvir
                    Eijab = eig[ib + ndns] + eig[ia + ndns] - eig[j] - eig[i]
                    Tijab[ijab] = -iajb[j, ia, i, ib] / Eijab
    return Tijab


def solve_Tijab(A_CSR, B, Tijab, p):
    Tijab, info = cg(A_CSR, B, x0=Tijab)
    return Tijab


def ext_koopmans(p, elag, n):
    elag_small = elag[: p.nbf5, : p.nbf5]
    nu = -np.einsum("qp,q,p->qp", elag_small, 1 / np.sqrt(n), 1 / np.sqrt(n))

    print("")
    print("---------------------------")
    print(" Extended Koopmans Theorem ")
    print("   Ionization Potentials   ")
    print("---------------------------")

    eigval, eigvec = np.linalg.eigh(nu)
    print(" OM        (eV)")
    print("---------------------------")
    for i, val in enumerate(eigval[::-1]):
        print("{: 3d}      {: 7.3f}".format(i, val * 27.2114))

    print("")
    print("EKT IP: {: 7.3f} eV".format(eigval[0] * 27.2114))
    print("")


def mulliken_pop(p, wfn, n, C, S):

    nPS = 2 * np.einsum("i,mi,ni,nm->m", n, C[:, : p.nbf5], C[:, : p.nbf5], S, optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += nPS[mu]

    print("")
    print("---------------------------------")
    print("  Mulliken Population Analysis   ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print(
            "{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(
                iatom, symbol, pop[iatom], wfn.molecule().Z(iatom) - pop[iatom]
            )
        )
    print("")


def lowdin_pop(p, wfn, n, C, S):

    S_12 = fractional_matrix_power(S, 0.5)

    S_12nPS_12 = 2 * np.einsum("sm,i,mi,ni,ns->s", S_12, n, C[:, : p.nbf5], C[:, : p.nbf5], S_12, optimize=True)

    pop = np.zeros((p.natoms))

    for mu in range(C.shape[0]):
        iatom = wfn.basisset().function_to_center(mu)
        pop[iatom] += S_12nPS_12[mu]

    print("")
    print("---------------------------------")
    print("   Lowdin Population Analysis    ")
    print("---------------------------------")
    print(" Idx  Atom   Population   Charge ")
    print("---------------------------------")
    for iatom in range(p.natoms):
        symbol = wfn.molecule().flabel(iatom)
        print(
            "{: 3d}    {:2s}    {: 5.2f}      {: 5.2f}".format(
                iatom, symbol, pop[iatom], wfn.molecule().Z(iatom) - pop[iatom]
            )
        )
    print("")


def M_diagnostic(p, n):

    m_vals = 2 * n
    if p.HighSpin:
        m_vals[p.nbeta : p.nalpha] = n[p.nbeta : p.nalpha]

    m_diagnostic = 0

    m_vals[p.no1 : p.nbeta] = 2.0 - m_vals[p.no1 : p.nbeta]

    if p.ndoc > 0:
        m_diagnostic += max(m_vals[p.no1 : p.nbeta])

    m_vals[p.nalpha : p.nbf5] = m_vals[p.nalpha : p.nbf5] - 0.0
    if p.ndoc > 0:
        m_diagnostic += max(m_vals[p.nalpha : p.nbf5])

    m_diagnostic = 0.5 * m_diagnostic

    print("")
    print("---------------------------------")
    print("   M Diagnostic: {:4.2f} ".format(m_diagnostic))
    print("---------------------------------")
    print("")


def matrix_product(x, M, V):

    w = x[0]
    y = x[1:]

    MM = M - w * V
    vec = np.matmul(MM, y)

    return np.linalg.norm(vec)


def ERPA(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, pp):

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis")
    print("---------------\n")

    tol_dn = 10**-3

    h = np.einsum("mi,mn,nj->ij", C[:, 0 : pp.nbf5], H, C[:, 0 : pp.nbf5], optimize=True)

    if pp.RI:
        b_pql = np.einsum("mnl,mp,nq->pql", b_mnl, C[:, 0 : pp.nbf5], C[:, 0 : pp.nbf5], optimize=True)
        I = np.einsum("pqR,slR->pqsl", b_pql, b_pql, optimize=True)
    else:
        I = np.einsum(
            "mp,nq,mnab,as,bl->pqsl",
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            I,
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            optimize=True,
        )
    if pp.gpu:
        I = I.get()

    c = np.sqrt(n)
    c[pp.no1 + pp.ndns :] *= -1

    I = np.einsum("rpsq->rspq", I, optimize=True)

    A = np.zeros((pp.nbf5, pp.nbf5, pp.nbf5, pp.nbf5))
    Id = np.identity(pp.nbf5)

    A += np.einsum("sq,pr,p->rspq", h, Id, n, optimize=True)
    A -= np.einsum("sq,pr,s->rspq", h, Id, n, optimize=True)
    A += np.einsum("pr,sq,q->rspq", h, Id, n, optimize=True)
    A -= np.einsum("pr,sq,r->rspq", h, Id, n, optimize=True)

    Daa, Dab = pynof.compute_2RDM(pp, n)

    time2 = time()

    ######################################
    A += np.einsum("stqu,purt->rspq", I, Daa, optimize=True)
    A -= np.einsum("stuq,purt->rspq", I, Daa, optimize=True)
    A += np.einsum("stqu,purt->rspq", I, Dab, optimize=True)
    A += np.einsum("stuq,putr->rspq", I, Dab, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Daa, optimize=True)
    A -= np.einsum("uprt,stqu->rspq", I, Daa, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Dab, optimize=True)
    A += np.einsum("uprt,stuq->rspq", I, Dab, optimize=True)

    time3 = time()

    ####
    A += np.einsum("pstu,turq->rspq", I, Daa, optimize=True)
    A -= np.einsum("pstu,utrq->rspq", I, Dab, optimize=True)
    A += np.einsum("tuqr,sptu->rspq", I, Daa, optimize=True)
    A -= np.einsum("tuqr,pstu->rspq", I, Dab, optimize=True)
    ####
    time4 = time()

    A += np.einsum("sq,tpwu,wurt->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("sq,tpwu,uwrt->rspq", Id, I, Dab, optimize=True)
    A += np.einsum("pr,tuwq,swtu->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("pr,tuwq,wstu->rspq", Id, I, Dab, optimize=True)

    M = np.zeros((pp.nbf5**2, pp.nbf5**2))

    # i is rows
    # j is columns

    time5 = time()

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time6 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time7 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q]
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p]
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p]

    v = np.zeros((pp.nbf5 * (pp.nbf5 - 1)))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = +(n[s] - n[r])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[s] - n[r])

    idx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn:
            idx.append(i)

    dim = (pp.nbf5) * (pp.nbf5 - 1) - len(idx)

    for i, j in enumerate(idx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format((pp.nbf5) * (pp.nbf5 - 1), dim, len(idx)))

    ######## ERPA0 ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff ApB {:3.1e} and Max AmB {:3.1e}".format(maxApBsym, maxAmBsym))

    MM = np.einsum("ij,jk,kl,lm->im", dNm1, ApB, dNm1, AmB, optimize=True)
    vals = np.linalg.eigvals(MM)
    vals = np.sqrt(vals)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    print("  Excitation energies ERPA0/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")
    vals_real = np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])

    sort_idx = np.argsort(vals_real)
    vals_real = vals_real[sort_idx]
    for i in range(min(10, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time8 = time()

    ######## ERPA  ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    CC = M[:dd, int(2 * dd) : int(2 * dd) + pp.nbf5]
    EE = M[int(2 * dd) : int(2 * dd) + pp.nbf5, :dd]
    FF = M[int(2 * dd) : int(2 * dd) + pp.nbf5, int(2 * dd) : int(2 * dd) + pp.nbf5]

    FFm1 = np.linalg.inv(FF)

    tmpMat = 2 * np.einsum("ij,jk,kl->il", CC, FFm1, EE, optimize=True)

    MM = np.einsum("ij,jk,kl,lm->im", dNm1, ApB - tmpMat, dNm1, AmB, optimize=True)
    vals = np.linalg.eigvals(MM)
    vals = np.sqrt(vals)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    print("  Excitation energies ERPA/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")
    vals_real = np.sort(
        np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])
    )
    for i in range(min(10, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time9 = time()

    ######## ERPA2 ########

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p] * 1 / 2 * (c[s] / (c[p] * (c[r] + c[s])))

    time10 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p] * 1 / 2 * (c[r] / (c[p] * (c[r] + c[s])))

    time11 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q] * (1 / c[r])
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p] * (1 / c[r])
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p] * (1 / (4 * c[p] * c[r]))

    v = np.zeros((pp.nbf5**2))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[r] - n[s])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = n[r] - n[s]
    for r in range(pp.nbf5):
        i += 1
        v[i] = 1

    V = np.zeros((pp.nbf5**2, pp.nbf5**2))
    np.fill_diagonal(V, v)

    lidx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn:
            lidx.append(i)

    dim = pp.nbf5**2 - len(lidx)

    for i, j in enumerate(lidx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    M_ERPA2 = M[:dim, :dim]

    time12 = time()

    M_ERPA2 = np.real(M_ERPA2)
    for i in range(dim):
        M_ERPA2[i, :] = M_ERPA2[i, :] / v[i]

    time13 = time()

    vals = np.linalg.eigvals(M_ERPA2)

    vals = vals * 27.2114
    vals_complex = np.array([val for val in vals if (np.abs(np.imag(val)) > 0.00000001)])

    time14 = time()

    print("  Excitation energies ERPA2/PNOF{}(eV)".format(pp.ipnof))
    print("  ==================================")
    vals_real = np.sort(
        np.array([np.real(val) for val in vals if (np.abs(np.imag(val)) < 0.00000001 and np.real(val) > 1.0)])
    )
    for i in range(min(11, len(vals_real))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals_real[i]))
    print("\n    Number of small diagonal elements of V moved to the end:", len(lidx))
    print("    Vals Complex: {}\n".format(np.size(vals_complex)))

    time15 = time()

    # Opt

    times = [
        time1,
        time2,
        time3,
        time4,
        time5,
        time6,
        time7,
        time8,
        time9,
        time10,
        time11,
        time12,
        time13,
        time14,
        time15,
    ]

    time_total = times[-1] - times[0]

    print("Δt            s              %")
    print("=================================")

    for i in range(len(times) - 1):
        d = times[i + 1] - times[i]
        print("t{j:>} - t{i:>}:   {d:3.1e}   ...   {p:4.1f}".format(i=i + 1, j=i + 2, d=d, p=d * 100 / time_total))


def iterative_ERPA0(wfn, mol, n, C, H, I, b_mnl, cj12, ck12, elag, number_of_excitations, pp):

    time1 = time()

    print("\n---------------")
    print(" ERPA Analysis")
    print("---------------\n")

    tol_dn = 10**-3

    h = np.einsum("mi,mn,nj->ij", C[:, 0 : pp.nbf5], H, C[:, 0 : pp.nbf5], optimize=True)

    if pp.RI:
        b_pql = np.einsum("mnl,mp,nq->pql", b_mnl, C[:, 0 : pp.nbf5], C[:, 0 : pp.nbf5], optimize=True)
        I = np.einsum("pqR,slR->pqsl", b_pql, b_pql, optimize=True)
    else:
        I = np.einsum(
            "mp,nq,mnab,as,bl->pqsl",
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            I,
            C[:, 0 : pp.nbf5],
            C[:, 0 : pp.nbf5],
            optimize=True,
        )
    if pp.gpu:
        I = I.get()

    c = np.sqrt(n)
    c[pp.no1 + pp.ndns :] *= -1

    I = np.einsum("rpsq->rspq", I, optimize=True)

    A = np.zeros((pp.nbf5, pp.nbf5, pp.nbf5, pp.nbf5))
    Id = np.identity(pp.nbf5)

    A += np.einsum("sq,pr,p->rspq", h, Id, n, optimize=True)
    A -= np.einsum("sq,pr,s->rspq", h, Id, n, optimize=True)
    A += np.einsum("pr,sq,q->rspq", h, Id, n, optimize=True)
    A -= np.einsum("pr,sq,r->rspq", h, Id, n, optimize=True)

    Daa, Dab = pynof.compute_2RDM(pp, n)

    time2 = time()

    ######################################
    A += np.einsum("stqu,purt->rspq", I, Daa, optimize=True)
    A -= np.einsum("stuq,purt->rspq", I, Daa, optimize=True)
    A += np.einsum("stqu,purt->rspq", I, Dab, optimize=True)
    A += np.einsum("stuq,putr->rspq", I, Dab, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Daa, optimize=True)
    A -= np.einsum("uprt,stqu->rspq", I, Daa, optimize=True)
    A += np.einsum("uptr,stqu->rspq", I, Dab, optimize=True)
    A += np.einsum("uprt,stuq->rspq", I, Dab, optimize=True)

    time3 = time()

    ####
    A += np.einsum("pstu,turq->rspq", I, Daa, optimize=True)
    A -= np.einsum("pstu,utrq->rspq", I, Dab, optimize=True)
    A += np.einsum("tuqr,sptu->rspq", I, Daa, optimize=True)
    A -= np.einsum("tuqr,pstu->rspq", I, Dab, optimize=True)
    ####
    time4 = time()

    A += np.einsum("sq,tpwu,wurt->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("sq,tpwu,uwrt->rspq", Id, I, Dab, optimize=True)
    A += np.einsum("pr,tuwq,swtu->rspq", Id, I, Daa, optimize=True)
    A -= np.einsum("pr,tuwq,wstu->rspq", Id, I, Dab, optimize=True)

    M = np.zeros((pp.nbf5**2, pp.nbf5**2))

    # i is rows
    # j is columns

    time5 = time()

    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time6 = time()

    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            j = -1
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, q, p]
            for q in range(pp.nbf5):
                for p in range(q + 1, pp.nbf5):
                    j += 1
                    M[i, j] = A[r, s, p, q]
            for p in range(pp.nbf5):
                j += 1
                M[i, j] = A[r, s, p, p]

    time7 = time()

    for r in range(pp.nbf5):
        i += 1
        j = -1
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, p, q]
        for q in range(pp.nbf5):
            for p in range(q + 1, pp.nbf5):
                j += 1
                M[i, j] = A[r, r, q, p]
        for p in range(pp.nbf5):
            j += 1
            M[i, j] = A[r, r, p, p]

    v = np.zeros((pp.nbf5 * (pp.nbf5 - 1)))
    i = -1
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = +(n[s] - n[r])
    for s in range(pp.nbf5):
        for r in range(s + 1, pp.nbf5):
            i += 1
            v[i] = -(n[s] - n[r])

    idx = []
    for i, vi in enumerate(v):
        if np.abs(vi) < tol_dn:
            idx.append(i)

    i_start = 0  # start of hono/virtual coupling after tol_dn
    for s in range(pp.ndoc - 1):
        for r in range(s + 1, pp.nbf5):
            if np.abs(v[i]) < tol_dn:
                i_start += 1

    dim = (pp.nbf5) * (pp.nbf5 - 1) - len(idx)

    for i, j in enumerate(idx):
        tmp = v[j - i]
        v[j - i : -1] = v[j - i + 1 :]
        v[-1] = tmp

        tmp = M[j - i, :].copy()
        M[j - i : -1, :] = M[j - i + 1 :, :]
        M[-1, :] = tmp
        tmp = M[:, j - i].copy()
        M[:, j - i : -1] = M[:, j - i + 1 :]
        M[:, -1] = tmp

    print("M_ERPA Orig dim: {} New dim: {} Elements below tol_dN: {}".format((pp.nbf5) * (pp.nbf5 - 1), dim, len(idx)))

    ######## ERPA0 ########

    dd = int(dim / 2)
    AA = M[:dd, :dd]
    BB = M[:dd, dd : int(2 * dd)]

    ApB = AA + BB
    AmB = AA - BB

    dA = np.zeros((dd, dd))
    np.fill_diagonal(dA, A.diagonal())

    dN = np.zeros((dd, dd))
    np.fill_diagonal(dN, v[:dd])
    dNm1 = np.linalg.pinv(dN)

    maxApBsym = np.max(np.abs(ApB - ApB.T))
    maxAmBsym = np.max(np.abs(AmB - AmB.T))
    print("Max diff ApB {:3.1e} and Max AmB {:3.1e}".format(maxApBsym, maxAmBsym))
    print()

    l = pp.nbf5 - pp.ndoc
    bp = np.zeros((dd, l))
    bm = np.zeros((dd, l))

    for j in range(l):
        bp[i_start + j, j] = 1
        bm[i_start + j, j] = 1

    bp = pynof.orthonormalize(bp, ApB)
    bm = pynof.orthonormalize(bm, AmB)

    w_old = np.zeros((number_of_excitations))
    for k in range(100):
        converged = True
        print("==== Iter {} ====".format(k))
        print("Number of guess vectors: ", np.shape(bp)[1])
        l = np.shape(bp)[1]

        S = np.einsum("ji,jk,kl->il", bm, dN, bp, optimize=True)

        lambda2, up = eigh(np.matmul(S.T, S))
        sort_idx = np.argsort(lambda2)[::-1]
        lambda2, up = lambda2[sort_idx], up[:, sort_idx]
        omega = 1 / np.sqrt(lambda2)
        um = np.einsum("i,ij,jk->ik", 1 / omega, S, up)
        for i, val in enumerate(omega[:number_of_excitations]):
            print(" Exc. en {}: {:6.3f}".format(i, val * 27.2114))

        if np.max(np.abs(w_old - omega[:number_of_excitations])) > 0.001:
            converged = False
            w_old = omega[:number_of_excitations]
        if converged:
            break

        for i in range(number_of_excitations):
            Rp = np.einsum("ij,jk,k->i", dN, bm[:, :l], um[:, i]) - 1 / omega[i] * np.einsum(
                "ij,jk,k->i", ApB, bp[:, :l], up[:, i]
            )
            Rm = np.einsum("ij,jk,k->i", dN, bp[:, :l], up[:, i]) - 1 / omega[i] * np.einsum(
                "ij,jk,k->i", AmB, bm[:, :l], um[:, i]
            )

            b_p1 = -np.matmul(
                np.linalg.pinv(1 / omega[i] ** 2 * np.matmul(dA, dA) - np.matmul(dN, dN)),
                1 / omega[i] * np.matmul(dA, Rp) + np.matmul(dN, Rm),
            )
            b_m1 = -np.matmul(
                np.linalg.pinv(1 / omega[i] ** 2 * np.matmul(dA, dA) - np.matmul(dN, dN)),
                1 / omega[i] * np.matmul(dA, Rm) + np.matmul(dN, Rp),
            )

            bp = np.hstack((bp, b_p1.reshape(np.shape(bp)[0], -1)))
            bm = np.hstack((bm, b_m1.reshape(np.shape(bm)[0], -1)))
        bp = pynof.orthonormalize(bp, ApB)
        bm = pynof.orthonormalize(bm, AmB)

        # relevancep = np.zeros((number_of_excitations))
        # relevancem = np.zeros((number_of_excitations))
        # for i in range(n_guess_vectors):
        #    relevancep[i] = np.max(np.abs(up[i,:number_of_excitations]))
        #    relevancem[i] = np.max(np.abs(um[i,:number_of_excitations]))
        # relevancep_idx = np.argsort(relevancep)[::-1]
        # relevancem_idx = np.argsort(relevancem)[::-1]
        # bp = bp[:,relevancep_idx]
        # bm = bm[:,relevancem_idx]
        # bp = bp[:,:10]
        # bm = bm[:,:10]

    vals = w_old * 27.2114

    print()
    print("  Excitation energies ERPA0/PNOF{}(eV)".format(pp.ipnof))
    print("  ===================================")

    for i in range(min(10, len(vals))):
        print("    Exc. en. {}: {:6.3f}".format(i, vals[i]))
