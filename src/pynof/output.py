import numpy as np


def fchk(filename, wfn, mol, jobtype, E_t, elag, n, C, p):

    C_new = C.copy()

    b = wfn.basisset()
    if b.has_puream():
        for ishell in range(b.nshell()):
            shell = b.shell(ishell)
            if shell.am == 1:
                idx = b.shell_to_basis_function(ishell)
                tmp_coeffs = C_new[idx : idx + 3, :].copy()
                C_new[idx + 0, :] = tmp_coeffs[1, :]
                C_new[idx + 1, :] = tmp_coeffs[2, :]
                C_new[idx + 2, :] = tmp_coeffs[0, :]
    elif not b.has_puream():
        for ishell in range(b.nshell()):
            shell = b.shell(ishell)
            if shell.am == 2:
                idx = b.shell_to_basis_function(ishell)
                tmp_coeffs = C_new[idx : idx + 6, :].copy()
                C_new[idx + 0, :] = tmp_coeffs[0, :]
                C_new[idx + 1, :] = tmp_coeffs[3, :]
                C_new[idx + 2, :] = tmp_coeffs[5, :]
                C_new[idx + 3, :] = tmp_coeffs[1, :] / np.sqrt(3)
                C_new[idx + 4, :] = tmp_coeffs[2, :] / np.sqrt(3)
                C_new[idx + 5, :] = tmp_coeffs[4, :] / np.sqrt(3)
            elif shell.am == 3:
                idx = b.shell_to_basis_function(ishell)
                tmp_coeffs = C_new[idx : idx + 10, :].copy()
                C_new[idx + 0, :] = tmp_coeffs[0, :]
                C_new[idx + 1, :] = tmp_coeffs[6, :]
                C_new[idx + 2, :] = tmp_coeffs[9, :]
                C_new[idx + 3, :] = tmp_coeffs[3, :] / np.sqrt(5)
                C_new[idx + 4, :] = tmp_coeffs[1, :] / np.sqrt(5)
                C_new[idx + 5, :] = tmp_coeffs[2, :] / np.sqrt(5)
                C_new[idx + 6, :] = tmp_coeffs[5, :] / np.sqrt(5)
                C_new[idx + 7, :] = tmp_coeffs[8, :] / np.sqrt(5)
                C_new[idx + 8, :] = tmp_coeffs[7, :] / np.sqrt(5)
                C_new[idx + 9, :] = tmp_coeffs[4, :] / np.sqrt(15)
            elif shell.am == 4:
                idx = b.shell_to_basis_function(ishell)
                tmp_coeffs = C_new[idx : idx + 15, :].copy()
                C_new[idx + 0, :] = tmp_coeffs[0, :]
                C_new[idx + 1, :] = tmp_coeffs[10, :]
                C_new[idx + 2, :] = tmp_coeffs[14, :]
                C_new[idx + 3, :] = tmp_coeffs[1, :] / np.sqrt(7)
                C_new[idx + 4, :] = tmp_coeffs[2, :] / np.sqrt(7)
                C_new[idx + 5, :] = tmp_coeffs[6, :] / np.sqrt(7)
                C_new[idx + 6, :] = tmp_coeffs[11, :] / np.sqrt(7)
                C_new[idx + 7, :] = tmp_coeffs[9, :] / np.sqrt(7)
                C_new[idx + 8, :] = tmp_coeffs[13, :] / np.sqrt(7)
                C_new[idx + 9, :] = tmp_coeffs[3, :] * np.sqrt(3) / np.sqrt(35)
                C_new[idx + 10, :] = tmp_coeffs[5, :] * np.sqrt(3) / np.sqrt(35)
                C_new[idx + 11, :] = tmp_coeffs[12, :] * np.sqrt(3) / np.sqrt(35)
                C_new[idx + 12, :] = tmp_coeffs[4, :] / np.sqrt(35)
                C_new[idx + 13, :] = tmp_coeffs[7, :] / np.sqrt(35)
                C_new[idx + 14, :] = tmp_coeffs[8, :] / np.sqrt(35)
            elif shell.am >= 5:
                print("Warning, angular momentum not supported in fchk")

    f = open(filename + ".fchk", "w")

    print("{}".format(filename), file=f)
    if p.ista == 0:
        print("{} PNOF{} {}".format(jobtype, p.ipnof, wfn.basisset().blend()), file=f)
    else:
        print("{} PNOF{}s {}".format(jobtype, p.ipnof, wfn.basisset().blend()), file=f)
    print("Number of atoms                            I           {:6d}".format(p.natoms), file=f)
    print("Charge                                     I           {:6d}".format(p.charge), file=f)
    print("Multiplicity                               I           {:6d}".format(p.mul), file=f)
    print("Number of electrons                        I           {:6d}".format(p.ne), file=f)
    print("Number of alpha electrons                  I           {:6d}".format(p.nalpha), file=f)
    print("Number of beta electrons                   I           {:6d}".format(p.nbeta), file=f)
    print("Number of basis functions                  I           {:6d}".format(p.nbf), file=f)
    print("Number of independant functions            I           {:6d}".format(p.nbf5), file=f)
    print("Number of contracted shells                I           {:6d}".format(wfn.basisset().nshell()), file=f)
    print("Highest angular momentum                   I           {:6d}".format(wfn.basisset().max_am()), file=f)
    print(
        "Largest degree of contraction              I           {:6d}".format(wfn.basisset().max_nprimitive()), file=f
    )
    print("Number of primitive shells                 I           {:6d}".format(wfn.basisset().nprimitive()), file=f)
    #    print("Virial Ratio                               R                {}".format(p.natoms),file=f)
    #    print("SCF Energy                                 R                {}".format(p.natoms),file=f)
    print("Atomic numbers                             I   N=      {:6d}".format(p.natoms), file=f)
    for i in range(p.natoms):
        Z = mol.Z(i)
        print(" {:11d}".format(int(Z)), end="", file=f)
        if (i + 1) % 6 == 0 or i + 1 == p.natoms:
            print("", file=f)
    print("Nuclear Charges                            R   N=      {:6d}".format(p.natoms), file=f)
    for i in range(p.natoms):
        Z = mol.Z(i)
        print(" {: .8e}".format(Z), end="", file=f)
        if (i + 1) % 6 == 0 or i + 1 == p.natoms:
            print("", file=f)
    print("Current cartesian coordinates              R   N=      {:6d}".format(p.natoms * 3), file=f)
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    idata = 0
    for xyz in coord:
        for ixyz in range(3):
            idata += 1
            print(" {: .8e}".format(xyz[ixyz]), end="", file=f)
            if idata % 5 == 0 or idata == p.natoms * 3:
                print("", file=f)
    print("Shell types                                I   N=      {:6d}".format(wfn.basisset().nshell()), file=f)
    for ishell in range(wfn.basisset().nshell()):
        if wfn.basisset().has_puream() and wfn.basisset().shell(ishell).am > 1:
            print(" {:11d}".format(-1 * wfn.basisset().shell(ishell).am), end="", file=f)
        else:
            print(" {:11d}".format(wfn.basisset().shell(ishell).am), end="", file=f)
        if (ishell + 1) % 6 == 0 or ishell + 1 == wfn.basisset().nshell():
            print("", file=f)
    print("Number of primitives per shell             I   N=      {:6d}".format(wfn.basisset().nshell()), file=f)
    for ishell in range(wfn.basisset().nshell()):
        print(" {:11d}".format(wfn.basisset().shell(ishell).nprimitive), end="", file=f)
        if (ishell + 1) % 6 == 0 or ishell + 1 == wfn.basisset().nshell():
            print("", file=f)
    print("Shell to atom map                          I   N=      {:6d}".format(wfn.basisset().nshell()), file=f)
    for ishell in range(wfn.basisset().nshell()):
        print(" {:11d}".format(wfn.basisset().shell(ishell).ncenter + 1), end="", file=f)
        if (ishell + 1) % 6 == 0 or ishell + 1 == wfn.basisset().nshell():
            print("", file=f)
    print("Coordinates of each shell                  R   N=      {:6d}".format(wfn.basisset().nshell() * 3), file=f)
    idata = 0
    for ishell in range(wfn.basisset().nshell()):
        xyz = coord[wfn.basisset().shell(ishell).ncenter]
        for ixyz in range(3):
            idata += 1
            print(" {: .8e}".format(xyz[ixyz]), end="", file=f)
            if idata % 5 == 0 or idata == wfn.basisset().nshell() * 3:
                print("", file=f)
    print("Total Energy                               R     {: .15e}".format(E_t), file=f)
    print("Primitive exponents                        R   N=      {:6d}".format(wfn.basisset().nprimitive()), file=f)
    idata = 0
    for ishell in range(wfn.basisset().nshell()):
        for iprim in range(wfn.basisset().shell(ishell).nprimitive):
            idata += 1
            print(" {: .8e}".format(wfn.basisset().shell(ishell).exp(iprim)), end="", file=f)
            if idata % 5 == 0 or idata == wfn.basisset().nprimitive():
                print("", file=f)
    print("Contraction coefficients                   R   N=      {:6d}".format(wfn.basisset().nprimitive()), file=f)
    idata = 0
    for ishell in range(wfn.basisset().nshell()):
        for iprim in range(wfn.basisset().shell(ishell).nprimitive):
            idata += 1
            print(" {: .8e}".format(wfn.basisset().shell(ishell).original_coef(iprim)), end="", file=f)
            if idata % 5 == 0 or idata == wfn.basisset().nprimitive():
                print("", file=f)

    e_val = elag[np.diag_indices(p.nbf)]

    print("Alpha Orbital Energies                     R   N=      {:6d}".format(p.nbf), file=f)
    for i in range(p.nbf):
        print(" {: .8e}".format(e_val[i]), end="", file=f)
        if (i + 1) % 5 == 0 or i + 1 == p.nbf:
            print("", file=f)
    #    print("Beta Orbital Energies                      R   N=      {:6d}".format(p.nbf),file=f)
    #    for i in range(p.nbf):
    #        print(" {: .8e}".format(e_val[i]),end ="", file=f)
    #        if((i+1)%5==0 or i+1==p.nbf):
    #            print("",file=f)
    print("Alpha MO coefficients                      R   N=      {:6d}".format(p.nbf * p.nbf), file=f)
    idata = 0
    for j in range(p.nbf):
        for i in range(p.nbf):
            idata += 1
            print(" {: .8e}".format(C_new[i, j]), end="", file=f)
            if idata % 5 == 0 or idata == int(p.nbf * p.nbf):
                print("", file=f)
    #    print("Beta MO coefficients                       R   N=       {:6d}".format(p.nbf*p.nbf),file=f)
    #    idata = 0
    #    for j in range(p.nbf):
    #        for i in range(p.nbf):
    #            idata += 1
    #            print(" {: .8e}".format(C_new[i,j]),end ="", file=f)
    #            if(idata%5==0 or idata==int(p.nbf*p.nbf)):
    #                print("",file=f)
    print("Total SCF Density                          R   N=       {:6d}".format(int(p.nbf * (p.nbf + 1) / 2)), file=f)
    DM = 2 * np.einsum("i,mi,ni->mn", n, C_new[:, : p.nbf5], C_new[:, : p.nbf5])
    idata = 0
    for mu in range(p.nbf):
        for nu in range(mu + 1):
            idata += 1
            print(" {: .8e}".format(DM[mu, nu]), end="", file=f)
            if idata % 5 == 0 or idata == int(p.nbf * (p.nbf + 1) / 2):
                print("", file=f)
    print("Natural orbital occupancies                R   N=       {:6d}".format(p.nbf5), file=f)
    for i in range(p.nbf5):
        print(" {: .8e}".format(n[i]), end="", file=f)
        if (i + 1) % 5 == 0 or i == p.nbf5:
            print("", file=f)

    f.close()
