import psi4
from scipy.linalg import orth
from numpy.linalg import matrix_power
import scipy
import numpy as np
import integrals
from scipy.linalg import eigh
import utils
import minimization
from numpy.linalg import lstsq
from scipy.linalg import orth


def find_orth(O, vec):
    A = np.hstack((O, vec.reshape((O.shape[0], 1))))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b, rcond=None)[0]


def project_MO(mol, C, old_basis):

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
    mints = psi4.core.MintsHelper(wfn.basisset())

    M = np.asarray(mints.ao_overlap(wfn.basisset(), old_basis))

    Sinv = mints.ao_overlap()
    Sm12 = mints.ao_overlap()
    Sp12 = mints.ao_overlap()
    Sinv.power(-1.0, 1.0e-14)
    Sm12.power(-0.5, 1.0e-14)
    Sp12.power(0.5, 1.0e-14)

    Cnew = np.matmul(np.matmul(Sinv, M), C)

    # Orthonormalize in C' = S^1/2 * C
    Cp = np.matmul(Sp12, Cnew)

    Cpnew = Cp[:, 0]
    Cpnew = Cpnew.reshape((Cpnew.shape[0], 1))

    for i in range(1, Cp.shape[1]):
        res = find_orth(Cpnew, Cp[:, i])
        res = res / np.linalg.norm(res)

        Cpnew = np.hstack((Cpnew, res.reshape((Cpnew.shape[0], 1))))

    C = np.matmul(Sm12, Cpnew)

    return Cnew


def complete_projection(wfn, mol, p, C, fmiug0, printmode=False):

    S, T, V, H, I, b_mnl = integrals.compute_integrals(wfn, mol, p)
    mints = psi4.core.MintsHelper(wfn.basisset())
    Sm12 = mints.ao_overlap()
    Sm12.power(-0.5, 1.0e-14)
    Sp12 = mints.ao_overlap()
    Sp12.power(0.5, 1.0e-14)

    # Energía Nuclear
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    E_i, Cguess = eigh(H, S)  # (HC = SCe)
    Cguess = utils.check_ortho(Cguess, S, p)

    # HF MO
    if p.hfidr:
        EHF, Cguess, fmiug0guess = minimization.hfidr(Cguess, H, I, b_mnl, E_nuc, p, printmode=True)

    # Combine projected MO
    Cguess[:, : C.shape[1]] = C[:, : C.shape[1]]

    # Orthonormalize in C' = S^1/2 * C
    Cp = np.matmul(Sp12, Cguess)

    Cpnew = Cp[:, 0]
    Cpnew = Cpnew.reshape((Cpnew.shape[0], 1))

    for i in range(1, Cp.shape[1]):
        res = find_orth(Cpnew, Cp[:, i])
        res = res / np.linalg.norm(res)

        Cpnew = np.hstack((Cpnew, res.reshape((Cpnew.shape[0], 1))))

    C = np.matmul(Sm12, Cpnew)

    C = utils.check_ortho(C, S, p)

    fmiug0guess[: fmiug0.shape[0]] = fmiug0
    fmiug0 = fmiug0guess

    return C, fmiug0
