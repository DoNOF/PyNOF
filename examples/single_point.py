import pynof

mol = pynof.molecule("""
0 1
  O  0.0000   0.000   0.116
  H  0.0000   0.749  -0.453
  H  0.0000  -0.749  -0.453
""")

p = pynof.param(mol,"cc-pvdz")

p.ipnof=8

p.RI = True
p.gpu = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p)
