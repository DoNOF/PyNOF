import psi4
import numpy as np
from scipy.linalg import eigh
from numpy.linalg import norm
from time import time
import pynof
from scipy.optimize import minimize
import functools
from libdlfind import dl_find
from libdlfind.callback import (dlf_get_gradient_wrapper,
                                dlf_put_coords_wrapper, make_dlf_get_params)

def optgeo(mol,p,C=None,n=None,fmiug0=None):
   
    wfn = p.wfn
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if(C is None or n is None or fmiug0 is None):
        E_t = pynof.compute_energy(mol,p,C,n,fmiug0,guess='HF',printmode=True)
    
    print("Initial Geometry (Bohrs)")
    print("========================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    coord = coord.flatten()
    res = minimize(energy_optgeo, coord, args=(symbols,p,True), jac=True, method='CG')

    coord = res.x

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

    E,grad = energy_optgeo(coord,symbols,p,printmode=True)

    coord = np.reshape(coord,(int(len(coord)/3),3))

    print("Final Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))
    print("Final Geometry (Angstroms)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0]*0.529177,xyz[1]*0.529177,xyz[2]*0.529177))

    return coord


def energy_optgeo(coord,symbols,p,printmode=False):

    coord = np.reshape(coord,(int(len(coord)/3),3))
    print("Iter Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    mol_string = "{} {} \n".format(p.charge,p.mul)
    for symbol,xyz in zip(symbols,coord):
        mol_string += "{:s} {} {} {}\n".format(symbol,xyz[0],xyz[1],xyz[2])
    mol_string += "units bohr\nnoreorient"
    mol = psi4.geometry(mol_string)
    
    # Paramdetros del sistema
    p.wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
  
    try:
        C,n,fmiug0 = pynof.read_all(p.title)
    except:
        C = pynof.read_C(p.title)
        n = pynof.read_n(p.title)
        fmiug0 = None

    #p.autozeros()
    p.autozeros(restart=True)
    
    t1 = time()
    E_t,C,n,fmiug0,grad = pynof.compute_energy(mol,p,C,n,fmiug0,gradients=True,printmode=printmode)
    #E_t,C,n,fmiug0 = pynof.compute_energy(mol,p,C,n,fmiug0,gradients=False,printmode=printmode)
    t2 = time()
    print("                       Total Energy:", E_t)

    print("====Gradient====")
    for i in range(p.natoms):
        print("Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(i,grad[i*3+0],grad[i*3+1],grad[i*3+2]))
    print("\n===Norm of the gradient===")
    print(norm(grad))
    
    return E_t,grad

def coords_to_string(symbols,coords,charge=0,mul=1):
    xyz = f"{charge} {mul} \n"
    for symbol, coord in zip(symbols, coords):
        xyz += f"{symbol} {coord[0]} {coord[1]} {coord[2]}\n"
    xyz += 'units bohr \n'
    return xyz

def calc_geo_energy(coords,symbols,p,C,n):
    
    print("Iter Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,coords):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))
    
    xyz=coords_to_string(symbols,coords)
    mol=pynof.molecule(xyz)
    p.wfn=psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('basis'))
    
    if (C is None) and (n is None):
        try:
            C,n=pynof.read_C(p.title),pynof.read_n(p.title)
        except FileNotFoundError:
            C,n=None,None
    else:
        pass
    
    p.autozeros(restart=True)
    energy,C,gamma,fmiug0,gradient = pynof.compute_energy(mol,p,C=C,n=n,gradients=True)
    
    print("                       Total Energy:", energy)

    print("====Gradient====")
    for i in range(p.natoms):
        print("Atom {:2d} {:10.4f} {:10.4f} {:10.4f}".format(i,gradient[i*3+0],gradient[i*3+1],gradient[i*3+2]))
    print("\n===Norm of the gradient===")
    print(norm(gradient))

    return energy,gradient*0.529177

# Function to call the energy and analytical gradients from PyNOF
@dlf_get_gradient_wrapper
def e_g_func(coordinates, iimage, kiter, calculator,p,symbols,C,n):

    energy, gradient = calculator(coordinates,symbols,p,C,n)

    return energy, gradient

# Function to store the results from DL-FIND
@dlf_put_coords_wrapper
def store_results(switch, energy, coordinates, iam,
                  traj_coords, traj_energies):
    traj_coords.append(np.array(coordinates))
    traj_energies.append(energy)
    return

def optgeo_dlfind(mol,p,C=None,n=None,fmiug0=None,dlf_get_params=None): 
    wfn = p.wfn
    coord, mass, symbols, Z, key = wfn.molecule().to_arrays()
    if(C is None or n is None or fmiug0 is None):
        E_t = pynof.compute_energy(mol,p,C,n,fmiug0,guess='HFIDr',printmode=True)

    print("Initial Geometry (Bohrs)")
    print("========================")
    for symbol,xyz in zip(symbols,coord):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))

    calculator = calc_geo_energy

    # List for storing the results
    traj_energies = []
    traj_coordinates = []

    if dlf_get_params == None:
        dlf_get_params = make_dlf_get_params(coords=coord)
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
    
    final_geometry=traj_coordinates[-1]
    niter=len(traj_energies)

    print("Final Geometry (Bohrs)")
    print("======================")
    for symbol,xyz in zip(symbols,final_geometry):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0],xyz[1],xyz[2]))
    print("Final Geometry (Angstroms)")
    print("======================")
    for symbol,xyz in zip(symbols,final_geometry):
        print("{:s} {:10.4f} {:10.4f} {:10.4f}".format(symbol,xyz[0]*0.529177,xyz[1]*0.529177,xyz[2]*0.529177)) 

    return coord
