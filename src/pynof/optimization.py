import functools
from time import time

import numpy as np
import psi4
from numpy.linalg import norm
from scipy.optimize import minimize

import pynof

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

def coords_to_string(symbols,coords,charge=0,mul=1):
    xyz = f"{charge} {mul} \n"
    for symbol, coord in zip(symbols, coords):
        xyz += f"{symbol} {coord[0]} {coord[1]} {coord[2]}\n"
    xyz += 'units bohr \n'
    return xyz

def optgeo(mol, p, C=None, n=None, fmiug0=None, method="CG", **minimize_kwargs):
   
    wfn = p.wfn
    coords, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if(C is None or n is None or fmiug0 is None):
        E_t = pynof.compute_energy(mol,p,C,n,fmiug0,guess='HF',printmode=True)
    
    _print_geometry(symbols, coords, "Initial Geometry (Bohrs)")

    coords = coords.flatten()
    res = minimize(energy_optgeo, coords, args=(symbols, p, True), jac=True, method=method,
                    **minimize_kwargs,)
    
    final_geometry = _reshape_coords(res.x)

    if(res.success):
        print("\n\n================¡Converged! :) ================\n\n")
    else:
        print(res.message)
        print('Status: ',res.status)
        print('Stored gradient: ',res.jac)
        print('Number of evaluations of the function: ',res.nfev)
        print('Number of evaluations of the Jacobian: ',res.njev)
        print('Number of evaluations performed by the optimizer: ',res.nit)

        print("\n\n================¡Not Converged! :( ================\n\n")

    energy_optgeo(final_geometry.reshape(-1),symbols,p,printmode=True)

    _print_geometry(symbols, final_geometry, "Final Geometry (Bohrs)")
    _print_geometry(symbols, final_geometry*0.529177, "Final Geometry (Angstroms)")

    return final_geometry


def energy_optgeo(coords,symbols,p,C=None,n=None,printmode=False):

    coords = _reshape_coords(coords)
    _print_geometry(symbols, coords, "Iter Geometry (Bohrs)")

    xyz=coords_to_string(symbols, coords, charge=p.charge, mul=p.mul)
    mol=pynof.molecule(xyz)    

    # Parametros del sistema
    p.wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
  
    try:
        C,n,fmiug0 = pynof.read_all(p.title)
    except:
        C = pynof.read_C(p.title)
        n = pynof.read_n(p.title)
        fmiug0 = None

    p.autozeros(restart=True)
    
    t1 = time()
    E_t,C,n,fmiug0,grad = pynof.compute_energy(mol,p,C,n,fmiug0,gradients=True,printmode=printmode)
    t2 = time()
    print("                       Total Energy:", E_t)

    print("====Gradient====")
    for i in range(p.natoms):
        print("Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(i,grad[i*3+0],grad[i*3+1],grad[i*3+2]))
    print("\n===Norm of the gradient===")
    print(norm(grad))
    
    return E_t,grad

try:
    from libdlfind import dl_find
    from libdlfind.callback import (
        dlf_get_gradient_wrapper,
        dlf_put_coords_wrapper,
        make_dlf_get_params,
    )

    # Function to call the energy and analytical gradients from PyNOF
    @dlf_get_gradient_wrapper
    def e_g_func(coords, iimage, kiter, calculator,p,symbols,C,n):
    
        energy, gradient = calculator(coords,symbols,p,C,n)
    
        return energy, gradient
    
    # Function to store the results from DL-FIND
    @dlf_put_coords_wrapper
    def store_results(switch, energy, coords, iam, traj_coords, traj_energies):
        traj_coords.append(np.array(coords))
        traj_energies.append(energy)
        return
except:
    pass
    
def optgeo_dlfind(mol,p,C=None,n=None,fmiug0=None,dlf_get_params=None): 
    wfn = p.wfn
    coords, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if(C is None or n is None or fmiug0 is None):
        E_t = pynof.compute_energy(mol,p,C,n,fmiug0,guess='HFIDr',printmode=True)

    _print_geometry(symbols, coords, "Initial Geometry (Bohrs)")

    calculator = energy_optgeo

    # List for storing the results
    traj_energies = []
    traj_coordinates = []

    try:

        if dlf_get_params == None:
            dlf_get_params = make_dlf_get_params(coords=coords)
    #    dlf_get_params = make_dlf_get_params(coords=coord,
    #                                         icoord=3,    # internal coordiantes
    #                                         ncons=ncons, # number of constraints
    #                                         spec=spec,   # array for constraints
    #                                         nconn=nconn,
    #                                         printl=6     # level of detail in prints of dlfind
    #                                         )
    
        e_g_func_ho = functools.partial(
                e_g_func,
                p = p,
                symbols=symbols,
                C=C,
                n=n,
                )
    
        dlf_get_gradient = functools.partial(e_g_func_ho,
                                             calculator=calculator)
        dlf_put_coords = functools.partial(
            store_results,
            traj_coords=traj_coordinates,
            traj_energies=traj_energies,
        )
    
        dl_find(
            nvarin=3*p.natoms,
            dlf_get_gradient=dlf_get_gradient,
            dlf_get_params=dlf_get_params,
            dlf_put_coords=dlf_put_coords,
        )

    except:
        print("\n\n\n")
        print("================== Error Message ===================")
        print("    DL-Find library is not installed. Please do:")
        print("    pip install libdlfind")
        print("====================================================")

    final_geometry=_reshape_coords(traj_coordinates[-1])
    niter=len(traj_energies)

    _print_geometry(symbols, final_geometry, "Final Geometry (Bohrs)")
    _print_geometry(symbols, final_geometry*0.529177, "Final Geometry (Angstroms)")
    
    return final_geometry
