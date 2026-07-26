import pynof

mol = pynof.molecule("""
0 1
    C     -8.597859    3.229765    0.000000
    H     -7.797863    2.981393    0.665745
    H     -8.275261    3.994097   -0.675742
    H     -8.883150    2.360420   -0.554751
    H     -9.435164    3.583151    0.564748
 """)

p = pynof.param(mol,"cc-pvdz")

p.ipnof = 8

p.RI = True
p.gpu = True
p.title = 'pynof'
#C,n=pynof.read_C(p.title),pynof.read_n(p.title)

pynof.dlfind_opt_geo(mol,p,spec=[1,1,1,1,1,1,1,1,1,1])
