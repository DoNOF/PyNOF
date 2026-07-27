import functools
from time import time

import numpy as np
import psi4
from numpy.linalg import norm
from scipy.optimize import minimize

import pynof
from libdlfind import dl_find
from libdlfind.callback import (
    dlf_get_gradient_wrapper,
    dlf_put_coords_wrapper,
    make_dlf_get_params,
)


def _reshape_coords(coords):
    arr = np.asarray(coords, dtype=float).reshape(-1)
    if arr.size % 3 != 0:
        raise ValueError("Coordinates must contain a multiple of 3 values")
    return arr.reshape(-1, 3)


def _print_geometry(symbols, coords, title):
    print(title)
    print("=" * len(title))
    for symbol, xyz in zip(symbols, coords):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol, xyz[0], xyz[1], xyz[2]))


def _coords_to_string(symbols, coords, charge=0, mul=1):
    xyz = f"{charge} {mul} \n"
    for symbol, coord in zip(symbols, coords):
        xyz += f"{symbol} {coord[0]} {coord[1]} {coord[2]}\n"
    xyz += "units bohr\nnoreorient\n"
    return xyz


def _build_molecule(symbols, coords, charge=0, mul=1):
    xyz = f"{charge} {mul} \n"
    for symbol, coord in zip(symbols, coords):
        xyz += f"{symbol} {coord[0]} {coord[1]} {coord[2]}\n"
    xyz += "units bohr\nnoreorient\n"
    return psi4.geometry(xyz)


def _build_wavefunction(mol, p):
    p.wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
    return p.wfn


def _print_geometry_summary(symbols, coords, title):
    _print_geometry(symbols, coords, title)
    print("Final Geometry (Angstroms)")
    print("======================")
    for symbol, xyz in zip(symbols, coords):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol, xyz[0] * 0.529177, xyz[1] * 0.529177, xyz[2] * 0.529177))


def optgeo(mol, p, C=None, n=None, fmiug0=None, method="CG", **minimize_kwargs):
    wfn = p.wfn
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if C is None or n is None or fmiug0 is None:
        pynof.compute_energy(mol, p, C, n, fmiug0, guess="HF", printmode=True)

    _print_geometry(symbols, coord, "Initial Geometry (Bohrs)")

    coord = _reshape_coords(coord).reshape(-1)
    res = minimize(
        energy_optgeo,
        coord,
        args=(symbols, p, True),
        jac=True,
        method=method,
        **minimize_kwargs,
    )

    final_coords = _reshape_coords(res.x)

    if res.success:
        print("\n\n================¡Converged! :) ================\n\n")
    else:
        print(res.message)
        print("Status: ", res.status)
        print("Stored gradient: ", res.jac)
        print("Number of evaluations of the function: ", res.nfev)
        print("Number of evaluations of the Jacobian: ", res.njev)
        print("Number of evaluations performed by the optimizer: ", res.nit)
        print("\n\n================¡Not Converged! :( ================\n\n")

    energy_optgeo(final_coords.reshape(-1), symbols, p, printmode=True)
    _print_geometry_summary(symbols, final_coords, "Final Geometry (Bohrs)")

    return final_coords


def energy_optgeo(coord, symbols, p, printmode=False):
    coord = _reshape_coords(coord)
    _print_geometry(symbols, coord, "Iter Geometry (Bohrs)")

    mol = _build_molecule(symbols, coord, charge=p.charge, mul=p.mul)
    _build_wavefunction(mol, p)

    try:
        C, n, fmiug0 = pynof.read_all(p.title)
    except Exception:
        C = pynof.read_C(p.title)
        n = pynof.read_n(p.title)
        fmiug0 = None

    p.autozeros(restart=True)

    t1 = time()
    E_t, C, n, fmiug0, grad = pynof.compute_energy(mol, p, C, n, fmiug0, gradients=True, printmode=printmode)
    t2 = time()
    print("                       Total Energy:", E_t)

    print("====Gradient====")
    for i in range(p.natoms):
        print("Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(i, grad[i * 3 + 0], grad[i * 3 + 1], grad[i * 3 + 2]))
    print("\n===Norm of the gradient===")
    print(norm(grad))

    return E_t, grad


def calc_geo_energy(coords, symbols, p, C, n):
    coords = _reshape_coords(coords)
    _print_geometry(symbols, coords, "Iter Geometry (Bohrs)")

    mol = _build_molecule(symbols, coords, charge=p.charge, mul=p.mul)
    _build_wavefunction(mol, p)

    if (C is None) and (n is None):
        try:
            C, n = pynof.read_C(p.title), pynof.read_n(p.title)
        except FileNotFoundError:
            C, n = None, None

    p.autozeros(restart=True)
    energy, C, gamma, fmiug0, gradient = pynof.compute_energy(mol, p, C=C, n=n, gradients=True)

    print("                       Total Energy:", energy)

    print("====Gradient====")
    for i in range(p.natoms):
        print(
            "Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(
                i, gradient[i * 3 + 0], gradient[i * 3 + 1], gradient[i * 3 + 2]
            )
        )
    print("\n===Norm of the gradient===")
    print(norm(gradient))

    return energy, gradient * 0.529177


@dlf_get_gradient_wrapper
def e_g_func(coordinates, iimage, kiter, calculator, p, symbols, C, n):
    energy, gradient = calculator(coordinates, symbols, p, C, n)
    return energy, gradient


@dlf_put_coords_wrapper
def store_results(switch, energy, coordinates, iam, traj_coords, traj_energies):
    traj_coords.append(np.array(coordinates))
    traj_energies.append(energy)
    return


def optgeo_dlfind(mol, p, C=None, n=None, fmiug0=None, dlf_get_params=None):
    wfn = p.wfn
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if C is None or n is None or fmiug0 is None:
        pynof.compute_energy(mol, p, C, n, fmiug0, guess="HFIDr", printmode=True)

    _print_geometry(symbols, coord, "Initial Geometry (Bohrs)")

    calculator = calc_geo_energy

    traj_energies = []
    traj_coordinates = []

    if dlf_get_params is None:
        dlf_get_params = make_dlf_get_params(coords=coord)

    e_g_func_ho = functools.partial(
        e_g_func,
        p=p,
        symbols=symbols,
        C=C,
        n=n,
    )

    dlf_get_gradient = functools.partial(e_g_func_ho, calculator=calculator)
    dlf_put_coords = functools.partial(
        store_results,
        traj_coords=traj_coordinates,
        traj_energies=traj_energies,
    )

    dl_find(
        nvarin=3 * p.natoms,
        dlf_get_gradient=dlf_get_gradient,
        dlf_get_params=dlf_get_params,
        dlf_put_coords=dlf_put_coords,
    )

    if traj_coordinates:
        final_geometry = _reshape_coords(traj_coordinates[-1])
    else:
        final_geometry = _reshape_coords(coord)

    _print_geometry_summary(symbols, final_geometry, "Final Geometry (Bohrs)")

    return final_geometry
