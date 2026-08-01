import pynof
from libdlfind.callback import make_dlf_get_params

mol = pynof.molecule(
"""
0 1
  O  
  H 1 0.96 
  H 1 0.96 2 97 
"""
)

p = pynof.param(mol, "cc-pvdz")

p.ipnof = 8

p.RI = True
p.gpu = True

p.title = 'h2o'

coords, mass, symbols, Z, key = p.wfn.molecule().to_arrays()
dlf_get_params = make_dlf_get_params(
    coords=coords,
    #                                         icoord=3,    # internal coordiantes
    #                                         ncons=ncons, # number of constraints
    #                                         spec=spec,   # array for constraints
    #                                         nconn=nconn,
    #                                         printl=6     # level of detail in prints of dlfind
)

C,n=pynof.read_C(p.title),pynof.read_n(p.title)

pynof.optgeo_dlfind(mol, p,C=C,n=n, dlf_get_params=dlf_get_params)
