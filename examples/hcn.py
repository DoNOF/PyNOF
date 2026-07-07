import pynof

mol = pynof.molecule("""
0 1
C
N  1 1.164
H  1 4.636 2 179.998
""")

p = pynof.param(mol,"def2-tzvpd")

p.ipnof = 8

p.RI = True
#p.gpu = True
p.title = 'hcn'

pynof.dlfind_opt_geo(mol,p,ncons=1,spec=[1,1,1,
                                         1,1,3,0,0,
                                         1,1,1])
