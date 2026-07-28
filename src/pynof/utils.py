import numpy as np
import pynof
from numba import prange, njit, jit
from time import time
import scipy as sp
from scipy.linalg import eigh, expm, solve
from scipy.optimize import root

try:
    import cupy as cp
except:
    pass


def computeF_RC_driver(J, K, n, H, cj12, ck12, p):

    if p.gpu:
        F = computeF_RC_GPU(J, K, n, H, cj12, ck12, p)
    else:
        F = computeF_RC_CPU(J, K, n, H, cj12, ck12, p)

    return F


def computeF_RC_CPU(J, K, n, H, cj12, ck12, p):

    # Matriz de Fock Generalizada
    F = np.zeros((p.nbf5, p.nbf, p.nbf))

    ini = 0
    if p.no1 > 1:
        ini = p.no1

    # nH
    F += np.einsum("i,mn->imn", n, H, optimize=True)  # i = [1,nbf5]

    # nJ
    F[ini : p.nbeta, :, :] += np.einsum(
        "i,imn->imn", n[ini : p.nbeta], J[ini : p.nbeta, :, :], optimize=True
    )  # i = [ini,nbeta]
    F[p.nalpha : p.nbf5, :, :] += np.einsum(
        "i,imn->imn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:, ini:], 0)  # Remove diag.
    F += np.einsum("ij,jmn->imn", cj12, J, optimize=True)  # i = [1,nbf5]
    # F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:, ini:], 0)  # Remove diag.
    F -= np.einsum("ij,jmn->imn", ck12, K, optimize=True)  # i = [1,nbf5]
    # F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    return F


def computeF_RC_GPU(J, K, n, H, cj12, ck12, p):

    # Matriz de Fock Generalizada
    F = cp.zeros((p.nbf5, p.nbf, p.nbf))

    ini = 0
    if p.no1 > 1:
        ini = p.no1

    # nH
    F += cp.einsum("i,mn->imn", n, H, optimize=True)  # i = [1,nbf5]

    # nJ
    F[ini : p.nbeta, :, :] += cp.einsum("i,imn->imn", n[ini : p.nbeta], J[ini : p.nbeta, :, :], optimize=True)
    F[p.nalpha : p.nbf5, :, :] += cp.einsum(
        "i,imn->imn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )

    # C^J J
    np.fill_diagonal(cj12[ini:, ini:], 0)  # Remove diag.
    F += cp.einsum("ij,jmn->imn", cj12, J, optimize=True)

    # -C^K K
    np.fill_diagonal(ck12[ini:, ini:], 0)  # Remove diag.
    F -= cp.einsum("ij,jmn->imn", ck12, K, optimize=True)

    return F.get()


def computeF_RO_driver(J, K, n, H, cj12, ck12, p):

    if p.gpu:
        F = computeF_RO_GPU(J, K, n, H, cj12, ck12, p)
    else:
        F = computeF_RO_CPU(J, K, n, H, cj12, ck12, p)

    return F


def computeF_RO_CPU(J, K, n, H, cj12, ck12, p):

    # Matriz de Fock Generalizada
    F = np.zeros((p.nbf5, p.nbf, p.nbf))

    ini = 0
    if p.no1 > 1:
        ini = p.no1

    # nH
    F[: p.nbeta, :, :] += np.einsum("i,mn->imn", n[: p.nbeta], H, optimize=True)  # i = [1,nbf5]
    F[p.nbeta : p.nalpha, :, :] += 0.5 * H  # i = [nbeta,nalpha]
    F[p.nalpha : p.nbf5, :, :] += np.einsum("i,mn->imn", n[p.nalpha : p.nbf5], H, optimize=True)  # i = [nalpha,nbf5]

    # nJ
    F[ini : p.nbeta, :, :] += np.einsum(
        "i,imn->imn", n[ini : p.nbeta], J[ini : p.nbeta, :, :], optimize=True
    )  # i = [ini,nbeta]
    F[p.nalpha : p.nbf5, :, :] += np.einsum(
        "i,imn->imn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:, ini:], 0)  # Remove diag.
    F[: p.nbeta, :, :] += np.einsum(
        "ij,jmn->imn", cj12[: p.nbeta, : p.nbeta], J[: p.nbeta, :, :], optimize=True
    )  # i = [1,nbeta]
    F[: p.nbeta, :, :] += np.einsum(
        "ij,jmn->imn", cj12[: p.nbeta, p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [1,nbeta]
    F[p.nalpha : p.nbf5, :, :] += np.einsum(
        "ij,jmn->imn", cj12[p.nalpha : p.nbf5, : p.nbeta], J[: p.nbeta, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    F[p.nalpha : p.nbf5, :, :] += np.einsum(
        "ij,jmn->imn", cj12[p.nalpha : p.nbf5, p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    # F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:, ini:], 0)  # Remove diag.
    F[: p.nbeta, :, :] -= np.einsum(
        "ij,jmn->imn", ck12[: p.nbeta, : p.nbeta], K[: p.nbeta, :, :], optimize=True
    )  # i = [1,nbeta]
    F[: p.nbeta, :, :] -= np.einsum(
        "ij,jmn->imn", ck12[: p.nbeta, p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [1,nbeta]
    F[p.nalpha : p.nbf5, :, :] -= np.einsum(
        "ij,jmn->imn", ck12[p.nalpha : p.nbf5, : p.nbeta], K[: p.nbeta, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    F[p.nalpha : p.nbf5, :, :] -= np.einsum(
        "ij,jmn->imn", ck12[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    # F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # SUMij
    F[: p.nbeta, :, :] += np.einsum(
        "i,jmn->imn", n[: p.nbeta], J[p.nbeta : p.nalpha, :, :] - 0.5 * K[p.nbeta : p.nalpha, :, :]
    )
    F[p.nbeta : p.nalpha, :, :] += 0.5 * np.einsum("jmn->mn", J[p.nbeta : p.nalpha, :, :] - K[p.nbeta : p.nalpha, :, :])
    F[p.nbeta : p.nalpha, :, :] -= 0.5 * (J[p.nbeta : p.nalpha, :, :] - K[p.nbeta : p.nalpha, :, :])  # Remove diag.
    F[p.nalpha : p.nbf5, :, :] += np.einsum(
        "i,jmn->imn", n[p.nalpha : p.nbf5], J[p.nbeta : p.nalpha, :, :] - 0.5 * K[p.nbeta : p.nalpha, :, :]
    )

    # PRODWROij
    F[p.nbeta : p.nalpha, :, :] += np.einsum("j,jmn->mn", n[: p.nbeta], J[: p.nbeta, :, :]) - 0.5 * np.einsum(
        "j,jmn->mn", n[: p.nbeta], K[: p.nbeta, :, :]
    )
    F[p.nbeta : p.nalpha, :, :] += np.einsum(
        "j,jmn->mn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :]
    ) - 0.5 * np.einsum("j,jmn->mn", n[p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :])

    return F


def computeF_RO_GPU(J, K, n, H, cj12, ck12, p):

    # Matriz de Fock Generalizada
    F = cp.zeros((p.nbf5, p.nbf, p.nbf))

    ini = 0
    if p.no1 > 1:
        ini = p.no1

    # nH
    F[: p.nbeta, :, :] += cp.einsum("i,mn->imn", n[: p.nbeta], H, optimize=True)  # i = [1,nbf5]
    F[p.nbeta : p.nalpha, :, :] += 0.5 * cp.array(H)  # i = [nbeta,nalpha]
    F[p.nalpha : p.nbf5, :, :] += cp.einsum("i,mn->imn", n[p.nalpha : p.nbf5], H, optimize=True)  # i = [nalpha,nbf5]

    # nJ
    F[ini : p.nbeta, :, :] += cp.einsum(
        "i,imn->imn", n[ini : p.nbeta], J[ini : p.nbeta, :, :], optimize=True
    )  # i = [ini,nbeta]
    F[p.nalpha : p.nbf5, :, :] += cp.einsum(
        "i,imn->imn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]

    # C^J J
    np.fill_diagonal(cj12[ini:, ini:], 0)  # Remove diag.
    F[: p.nbeta, :, :] += cp.einsum(
        "ij,jmn->imn", cj12[: p.nbeta, : p.nbeta], J[: p.nbeta, :, :], optimize=True
    )  # i = [1,nbeta]
    F[: p.nbeta, :, :] += cp.einsum(
        "ij,jmn->imn", cj12[: p.nbeta, p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [1,nbeta]
    F[p.nalpha : p.nbf5, :, :] += cp.einsum(
        "ij,jmn->imn", cj12[p.nalpha : p.nbf5, : p.nbeta], J[: p.nbeta, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    F[p.nalpha : p.nbf5, :, :] += cp.einsum(
        "ij,jmn->imn", cj12[p.nalpha : p.nbf5, p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    # F[ini:p.nbf5,:,:] -= np.einsum('ii,imn->imn',cj12[ini:p.nbf5,ini:p.nbf5],J[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # -C^K K
    np.fill_diagonal(ck12[ini:, ini:], 0)  # Remove diag.
    F[: p.nbeta, :, :] -= cp.einsum(
        "ij,jmn->imn", ck12[: p.nbeta, : p.nbeta], K[: p.nbeta, :, :], optimize=True
    )  # i = [1,nbeta]
    F[: p.nbeta, :, :] -= cp.einsum(
        "ij,jmn->imn", ck12[: p.nbeta, p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [1,nbeta]
    F[p.nalpha : p.nbf5, :, :] -= cp.einsum(
        "ij,jmn->imn", ck12[p.nalpha : p.nbf5, : p.nbeta], K[: p.nbeta, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    F[p.nalpha : p.nbf5, :, :] -= cp.einsum(
        "ij,jmn->imn", ck12[p.nalpha : p.nbf5, p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :], optimize=True
    )  # i = [nalpha,nbf5]
    # F[ini:p.nbf5,:,:] += np.einsum('ii,imn->imn',ck12[ini:p.nbf5,ini:p.nbf5],K[ini:p.nbf5,:,:],optimize=True) # quita i==j

    # SUMij
    F[: p.nbeta, :, :] += cp.einsum(
        "i,jmn->imn", n[: p.nbeta], J[p.nbeta : p.nalpha, :, :] - 0.5 * K[p.nbeta : p.nalpha, :, :]
    )
    F[p.nbeta : p.nalpha, :, :] += 0.5 * cp.einsum("jmn->mn", J[p.nbeta : p.nalpha, :, :] - K[p.nbeta : p.nalpha, :, :])
    F[p.nbeta : p.nalpha, :, :] -= 0.5 * cp.array(
        (J[p.nbeta : p.nalpha, :, :] - K[p.nbeta : p.nalpha, :, :])
    )  # Remove diag.
    F[p.nalpha : p.nbf5, :, :] += cp.einsum(
        "i,jmn->imn", n[p.nalpha : p.nbf5], J[p.nbeta : p.nalpha, :, :] - 0.5 * K[p.nbeta : p.nalpha, :, :]
    )

    # PRODWROij
    F[p.nbeta : p.nalpha, :, :] += cp.einsum("j,jmn->mn", n[: p.nbeta], J[: p.nbeta, :, :]) - 0.5 * cp.einsum(
        "j,jmn->mn", n[: p.nbeta], K[: p.nbeta, :, :]
    )
    F[p.nbeta : p.nalpha, :, :] += cp.einsum(
        "j,jmn->mn", n[p.nalpha : p.nbf5], J[p.nalpha : p.nbf5, :, :]
    ) - 0.5 * cp.einsum("j,jmn->mn", n[p.nalpha : p.nbf5], K[p.nalpha : p.nbf5, :, :])

    return F.get()


def computeLagrange(F, C, p):

    G = np.einsum("imn,ni->mi", F, C[:, 0 : p.nbf5], optimize=True)

    # Compute Lagrange multipliers
    elag = np.zeros((p.nbf, p.nbf))
    elag[0 : p.noptorb, 0 : p.nbf5] = np.einsum("mi,mj->ij", C[:, 0 : p.noptorb], G, optimize=True)[
        0 : p.noptorb, 0 : p.nbf5
    ]

    return elag


def computeLagrange2(n, cj12, ck12, C, H, I, b_mnl, p):

    if p.RI:
        Hmat, b_MO = pynof.JKH_MO_tmp(C, H, I, b_mnl, p)
    else:
        Hmat, I_MO = pynof.JKH_MO_tmp(C, H, I, b_mnl, p)

    np.fill_diagonal(cj12, 0)  # Remove diag.
    np.fill_diagonal(ck12, 0)  # Remove diag.

    if p.gpu:
        elag = cp.zeros((p.nbf, p.nbf))
        n = cp.array(n)
        cj12 = cp.array(cj12)
        ck12 = cp.array(ck12)
        if p.RI:
            if p.MSpin == 0:
                # 2ndH/dy_ab
                elag[:, : p.nbf5] += cp.einsum("b,ab->ab", n, Hmat[:, : p.nbf5], optimize=True)

                # dJ_pp/dy_ab
                elag[:, : p.nbeta] += cp.einsum(
                    "b,abk,bbk->ab", n[: p.nbeta], b_MO[:, : p.nbeta, :], b_MO[: p.nbeta, : p.nbeta, :], optimize=True
                )
                elag[:, p.nalpha : p.nbf5] += cp.einsum(
                    "b,abk,bbk->ab",
                    n[p.nalpha : p.nbf5],
                    b_MO[:, p.nalpha : p.nbf5, :],
                    b_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5, :],
                    optimize=True,
                )

                # C^J_pq dJ_pq/dy_ab
                tmp = cp.einsum("bq,qqk->bk", cj12, b_MO[: p.nbf5, : p.nbf5, :], optimize=True)
                elag[:, : p.nbf5] += cp.einsum("abk,bk->ab", b_MO[:, : p.nbf5, :], tmp, optimize=True)

                # -C^K_pq dK_pq/dy_ab
                elag[:, : p.nbf5] += -cp.einsum(
                    "bq,aqk,bqk->ab", ck12, b_MO[:, : p.nbf5, :], b_MO[: p.nbf5, : p.nbf5, :], optimize=True
                )
        else:
            if p.MSpin == 0:
                # 2ndH/dy_ab
                elag[:, : p.nbf5] += cp.einsum("b,ab->ab", n, Hmat[:, : p.nbf5], optimize=True)

                # dJ_pp/dy_ab
                elag[:, : p.nbeta] += cp.einsum(
                    "b,abbb->ab", n[: p.nbeta], I_MO[:, : p.nbeta, : p.nbeta, : p.nbeta], optimize=True
                )
                elag[:, p.nalpha : p.nbf5] += cp.einsum(
                    "b,abbb->ab",
                    n[p.nalpha : p.nbf5],
                    I_MO[:, p.nalpha : p.nbf5, p.nalpha : p.nbf5, p.nalpha : p.nbf5],
                    optimize=True,
                )

                # C^J_pq dJ_pq/dy_ab
                elag[:, : p.nbf5] += cp.einsum(
                    "bq,abqq->ab", cj12, I_MO[:, : p.nbf5, : p.nbf5, : p.nbf5], optimize=True
                )

                # -C^K_pq dK_pq/dy_ab
                elag[:, : p.nbf5] += -cp.einsum(
                    "bq,aqbq->ab", ck12, I_MO[:, : p.nbf5, : p.nbf5, : p.nbf5], optimize=True
                )
        return elag.get(), Hmat.get()
    else:
        elag = np.zeros((p.nbf, p.nbf))
        grad = np.zeros((p.nbf, p.nbf))
        if p.RI:
            if p.MSpin == 0:
                # 2ndH/dy_ab
                elag[:, : p.nbf5] += np.einsum("b,ab->ab", n, Hmat[:, : p.nbf5], optimize=True)

                # dJ_pp/dy_ab
                elag[:, : p.nbeta] += np.einsum(
                    "b,abk,bbk->ab", n[: p.nbeta], b_MO[:, : p.nbeta, :], b_MO[: p.nbeta, : p.nbeta, :], optimize=True
                )
                elag[:, p.nalpha : p.nbf5] += np.einsum(
                    "b,abk,bbk->ab",
                    n[p.nalpha : p.nbf5],
                    b_MO[:, p.nalpha : p.nbf5, :],
                    b_MO[p.nalpha : p.nbf5, p.nalpha : p.nbf5, :],
                    optimize=True,
                )

                # C^J_pq dJ_pq/dy_ab
                tmp = np.einsum("bq,qqk->bk", cj12, b_MO[: p.nbf5, : p.nbf5, :], optimize=True)
                elag[:, : p.nbf5] += np.einsum("abk,bk->ab", b_MO[:, : p.nbf5, :], tmp, optimize=True)

                # -C^K_pq dK_pq/dy_ab
                elag[:, : p.nbf5] += -np.einsum(
                    "bq,aqk,bqk->ab", ck12, b_MO[:, : p.nbf5, :], b_MO[: p.nbf5, : p.nbf5, :], optimize=True
                )
        else:
            if p.MSpin == 0:
                # 2ndH/dy_ab
                elag[:, : p.nbf5] += np.einsum("b,ab->ab", n, Hmat[:, : p.nbf5], optimize=True)

                # dJ_pp/dy_ab
                elag[:, : p.nbeta] += np.einsum(
                    "b,abbb->ab", n[: p.nbeta], I_MO[:, : p.nbeta, : p.nbeta, : p.nbeta], optimize=True
                )
                elag[:, p.nalpha : p.nbf5] += np.einsum(
                    "b,abbb->ab",
                    n[p.nalpha : p.nbf5],
                    I_MO[:, p.nalpha : p.nbf5, p.nalpha : p.nbf5, p.nalpha : p.nbf5],
                    optimize=True,
                )

                # C^J_pq dJ_pq/dy_ab
                elag[:, : p.nbf5] += np.einsum(
                    "bq,abqq->ab", cj12, I_MO[:, : p.nbf5, : p.nbf5, : p.nbf5], optimize=True
                )

                # -C^K_pq dK_pq/dy_ab
                elag[:, : p.nbf5] += -np.einsum(
                    "bq,aqbq->ab", ck12, I_MO[:, : p.nbf5, : p.nbf5, : p.nbf5], optimize=True
                )

    return elag, Hmat


def computeE_elec(H, C, n, elag, p):
    # EELECTRr
    E = 0

    E = E + np.einsum("ii", elag[: p.nbf5, : p.nbf5], optimize=True)
    E = E + np.einsum("i,mi,mn,ni", n[: p.nbeta], C[:, : p.nbeta], H, C[:, : p.nbeta], optimize=True)
    if not p.HighSpin:
        E = E + np.einsum(
            "i,mi,mn,ni", n[p.nbeta : p.nalpha], C[:, p.nbeta : p.nalpha], H, C[:, p.nbeta : p.nalpha], optimize=True
        )
    elif p.HighSpin:
        E = E + 0.5 * np.einsum("mi,mn,ni", C[:, p.nbeta : p.nalpha], H, C[:, p.nbeta : p.nalpha], optimize=True)

    E = E + np.einsum(
        "i,mi,mn,ni", n[p.nalpha : p.nbf5], C[:, p.nalpha : p.nbf5], H, C[:, p.nalpha : p.nbf5], optimize=True
    )

    return E


def computeLagrangeConvergency(elag):
    # Convergency

    sumdiff = np.sum(np.abs(elag - elag.T))
    maxdiff = np.max(np.abs(elag - elag.T))

    return sumdiff, maxdiff


def ENERGY1r(C, n, H, I, b_mnl, cj12, ck12, p):

    if p.no1 == 0:
        elag, Hmat = pynof.computeLagrange2(n, cj12, ck12, C, H, I, b_mnl, p)
        E = np.einsum("ii", elag[: p.nbf5, : p.nbf5], optimize=True)
        E = E + np.einsum("i,ii", n[: p.nbf5], Hmat[: p.nbf5, : p.nbf5], optimize=True)
    else:
        J, K = pynof.computeJKj(C, I, b_mnl, p)
        if p.MSpin == 0:
            F = computeF_RC_driver(J, K, n, H, cj12, ck12, p)
        elif not p.MSpin == 0:
            F = computeF_RO_driver(J, K, n, H, cj12, ck12, p)
        elag = computeLagrange(F, C, p)
        E = computeE_elec(H, C, n, elag, p)

    sumdiff, maxdiff = computeLagrangeConvergency(elag)

    return E, elag, sumdiff, maxdiff


def fmiug_scaling(fmiug0, elag, i_ext, nzeros, nbf, noptorb):

    # scaling
    fmiug = np.zeros((nbf, nbf))
    if i_ext == 0 and fmiug0 is None:
        fmiug[:noptorb, :noptorb] = (elag[:noptorb, :noptorb] + elag[:noptorb, :noptorb].T) / 2

    else:
        fmiug[:noptorb, :noptorb] = elag[:noptorb, :noptorb] - elag[:noptorb, :noptorb].T
        fmiug = np.tril(fmiug, -1) + np.tril(fmiug, -1).T
        for k in range(nzeros + 9 + 1):
            fmiug[(abs(fmiug) > 10 ** (9 - k)) & (abs(fmiug) < 10 ** (10 - k))] *= 0.1
        np.fill_diagonal(fmiug[:noptorb, :noptorb], fmiug0[:noptorb])

    return fmiug


@njit
def fmiug_diis(fk, fmiug, idiis, bdiis, cdiis, maxdiff, noptorb, ndiis, perdiis):

    fk[idiis, 0:noptorb, 0:noptorb] = fmiug[0:noptorb, 0:noptorb]
    for m in range(idiis + 1):
        bdiis[m][idiis] = 0
        for i in range(noptorb):
            for j in range(i):
                bdiis[m][idiis] = bdiis[m][idiis] + fk[m][i][j] * fk[idiis][j][i]
        bdiis[idiis][m] = bdiis[m][idiis]
        bdiis[m][idiis + 1] = -1.0
        bdiis[idiis + 1][m] = -1.0
    bdiis[idiis + 1][idiis + 1] = 0.0

    if idiis >= ndiis:
        cdiis = np.zeros((idiis + 2))
        cdiis[0 : idiis + 1] = 0.0
        cdiis[idiis + 1] = -1.0
        x = np.linalg.solve(bdiis[0 : idiis + 2, 0 : idiis + 2], cdiis[0 : idiis + 2])

        for i in range(noptorb):
            for j in range(i):
                fmiug[i][j] = 0.0
                for k in range(idiis + 1):
                    fmiug[i][j] = fmiug[i][j] + x[k] * fk[k][i][j]
                fmiug[j][i] = fmiug[i][j]
    #        print("idiis", idiis)
    if idiis >= ndiis and perdiis:
        idiis = 0
    else:
        idiis = idiis + 1

    return fk, fmiug, idiis, bdiis


def check_ortho(C, S, p):

    # Revisa ortonormalidad
    orthonormality = True
    CTSC = np.matmul(np.matmul(np.transpose(C), S), C)
    ortho_deviation = np.abs(CTSC - np.identity(p.nbf))
    if np.any(ortho_deviation > 10**-6):
        orthonormality = False
    if not orthonormality:
        print(
            "Orthonormality violations {:d}, Maximum Violation {:f}".format(
                (ortho_deviation > 10**-6).sum(), ortho_deviation.max()
            )
        )
        print("Trying to orthonormalize")
        C = orthonormalize(C, S)
        C = check_ortho(C, S, p)
    else:
        print("No violations of the orthonormality")
    for j in range(p.nbf):
        # Obtiene el índice del coeficiente con mayor valor absoluto del MO
        idxmaxabsval = 0
        for i in range(p.nbf):
            if abs(C[i][j]) > abs(C[idxmaxabsval][j]):
                idxmaxabsval = i
    # Ajusta el signo del MO
    sign = np.sign(C[idxmaxabsval][j])
    C[0 : p.nbf, j] = sign * C[0 : p.nbf, j]

    return C


def orthonormalize(C, S):
    eigval, eigvec = eigh(S)
    S_12 = np.einsum("ij,j->ij", eigvec, eigval ** (-1 / 2), optimize=True)

    Cnew = np.einsum("ik,kj->ij", S, C, optimize=True)

    Cnew2 = np.einsum("ki,kj->ij", S_12, Cnew)

    for i in range(Cnew2.shape[1]):
        norm = np.einsum("k,k->", Cnew2[:, i], Cnew2[:, i], optimize=True)
        Cnew2[:, i] = Cnew2[:, i] / np.sqrt(norm)
        for j in range(i + 1, Cnew2.shape[1]):
            val = -np.einsum("k,k->", Cnew2[:, i], Cnew2[:, j], optimize=True)
            Cnew2[:, j] = Cnew2[:, j] + val * Cnew2[:, i]

    C = np.einsum("ik,kj->ij", S_12, Cnew2, optimize=True)

    return C


def orthonormalize2(M):

    for i in range(M.shape[1]):
        norm = np.einsum("k,k->", M[:, i], M[:, i], optimize=True)
        M[:, i] = M[:, i] / np.sqrt(norm)
        for j in range(i + 1, M.shape[1]):
            val = -np.einsum("k,k->", M[:, i], M[:, j], optimize=True)
            M[:, j] = M[:, j] + val * M[:, i]

    return M


def rotate_orbital(y, C, p):

    ynew = np.zeros((p.nbf, p.nbf))

    n = 0
    for i in range(p.nbf5):
        for j in range(i + 1, p.nbf):
            ynew[i, j] = y[n]
            ynew[j, i] = -y[n]
            n += 1

    U = expm(ynew)

    Cnew = np.einsum("mr,rp->mp", C, U, optimize=True)

    return Cnew


@njit(cache=True)
def extract_tiu_tensor(t, k):
    dim = len(t)
    var = int(dim * (dim - 1) / 2)
    t_extracted = np.zeros((var, var))
    i = 0
    for p in range(dim):
        for q in range(p + k, dim):
            j = 0
            for r in range(dim):
                for s in range(r + k, dim):
                    t_extracted[i, j] = t[p, q, r, s]
                    j += 1
            i += 1

    return t_extracted


def perturb_solution(C, gamma, grad_orb, grad_occ, p):
    grad_orb = pynof.perturb_gradient(grad_orb, p.tol_gorb)
    y = -grad_orb
    C = pynof.rotate_orbital(y, C, p)
    grad_occ = pynof.perturb_gradient(grad_occ, p.tol_gocc)
    gamma += -grad_occ

    return C, gamma


@njit
def perturb_gradient(grad, tol):
    dim = grad.shape[0]
    for i in range(dim):
        if np.abs(grad[i]) < tol:
            grad[i] = np.sign(grad[i]) * tol

    return grad


@njit(cache=True)
def n_to_gammas_trigonometric(n, no1, ndoc, ndns, ncwo):
    """Transform n to gammas in the trigonometric encoding"""

    nv = ncwo * ndoc
    gamma = np.zeros(nv)
    for i in range(ndoc):
        idx = no1 + i
        gamma[i] = np.arccos(np.sqrt(2.0 * n[idx] - 1.0))
        prefactor = max(1 - n[idx], 1e-14)
        for j in range(ncwo - 1):
            jg = ndoc + (ndoc - i - 1) +  j * ndoc
            ig = no1 + ndns + (ndoc - i - 1) + j * ndoc
            gamma[jg] = np.arcsin(np.sqrt(n[ig] / prefactor))
            prefactor = prefactor * (np.cos(gamma[jg])) ** 2
    return gamma

@njit(cache=True)
def n_to_gammas_softmax(n, no1, ndoc, ndns, ncwo):
    """Transform n to gammas in the softmax encoding"""

    nv = ncwo * ndoc
    gamma = np.zeros(nv)
    for i in range(ndoc):

        ll = no1 + ndns + (ndoc - i - 1)
        ul = ll + ndoc * (ncwo - 1)

        llg = ll - ndns - no1
        ulg = ul - ndns - no1

        n_pi = n[ll:ul:ndoc]
        gamma_pi = gamma[llg:ulg:ndoc]

        for j in range(ncwo):
            gamma_pi[j] = np.log(max(n_pi[j], 1e-16)) - np.log(n[i])

    return gamma

def n_to_gammas_ebi(n):
    """Transform n to gammas in the ebi encoding

    x_p = erf^-1 (2n_p - 1)

    """

    nv = np.shape(n)[0]
    gamma = np.zeros((nv))
    for i in range(nv):
        gamma[i] = sp.special.erfinv(2 * n[i] - 1)

    return gamma


@njit(cache=True)
def compute_gammas_trigonometric(ndoc, ncwo):
    """Compute guess for occupation numbers according to
    the trigonometric encoding"""

    nv = ncwo * ndoc

    gamma = np.zeros((nv))
    for i in range(ndoc):
        gamma[i] = np.arccos(np.sqrt(2.0 * 0.999 - 1.0))
        for j in range(ncwo - 1):
            ig = ndoc + (ndoc - i - 1) + j * ncwo
            gamma[ig] = np.arcsin(np.sqrt(1.0 / (ncwo - j)))
    return gamma

@njit(cache=True)
def compute_gammas_softmax(ndoc, ncwo):
    """Compute a guess for gammas in the softmax parameterization
    of the occupation numbers"""

    # TODO: Add equations and look to reduce a variable

    nv = ncwo * ndoc

    gamma = np.zeros((nv))
    for i in range(ndoc):
        for j in range(ncwo):
            ig = (ndoc - i - 1) + j * ncwo
            gamma[ig] = np.log(max(0.001 / ncwo, 1e-16)) - np.log(0.999)
    return gamma

# @njit(jit=True,cache=True)
def compute_gammas_ebi(ndoc, nbf):
    """Compute a guess for gammas in the softmax parameterization
    of the occupation numbers"""

    # TODO: Add equations and look to reduce a variable

    nv = nbf
    gamma = np.zeros((nv))
    for i in range(ndoc):
        gamma[i] = sp.special.erfinv(2 * 0.999 - 1)
    val = ndoc * (1.0 - 0.999)
    for i in range(ndoc, nv):
        gamma[i] = sp.special.erfinv(2 * val / nv - 1)

    return gamma
