import pynof

mol = pynof.molecule("""
0 1
  O  
  H 1 0.96 
  H 1 0.96 2 97 
""")

p = pynof.param(mol,"cc-pvdz")

p.ipnof = 8

p.RI = True
p.gpu = True

dlf_get_params = make_dlf_get_params(coords=coord,
#                                         icoord=3,    # internal coordiantes
#                                         ncons=ncons, # number of constraints
#                                         spec=spec,   # array for constraints
#                                         nconn=nconn,
#                                         printl=6     # level of detail in prints of dlfind
                                         )


pynof.optgeo_dlfind(mol,p,dlf_get_params=dlf_get_params)
