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

pynof.optgeo_dlfind(mol,p)
