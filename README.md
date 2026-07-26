# PyNOF

| **Citation** | [![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jchemed.5c01470-blue)](https://doi.org/10.1021/acs.jchemed.5c01470) |
| ------------ | ---------------------------------------------------------------------------------------------- |
| **Citation** | [![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0316927-blue)](https://doi.org/10.1063/5.0316927) |

Formally read as Python-Natural-Orbital-Functionals, PyNOF is based on the original [DoNOF](https://github.com/DoNOF/DoNOFsw/) software written in Fortran by Prof. Mario Piris, but it takes advantage of the Python capabilities such as optimizers, vectorization via numpy and gpu compatibility via cupy.

# <img src="https://github.com/felipelewyee/PyNOF/blob/master/PyNOF.png" height=150>

# :rocket: Installation

We recommend to perform the installation inside an [Anaconda](https://www.anaconda.com/) enviroment:
~~~
conda create -y -n pynof
conda activate pynof
~~~

PyNOF uses [Psi4](https://psicode.org/installs/latest) for integrals, so it is necessary to install it first:
~~~
conda install -y psi4 -c conda-forge/label/libint_dev -c conda-forge
~~~
then, you can simply install PyNOF using pip
~~~
pip install pynof
~~~

[Optional] Integrals transformations can benefit from a GPU. If available, just install [cupy](https://cupy.dev/)
~~~
conda install -y -c conda-forge cupy
~~~

# :sunrise: Example

A pynof input has the following parts:
- Import pynof
- A declaration of the molecule geometry
- A parameter object
- The calculation instruction

~~~
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
#p.gpu = True

E,C,gamma,fmiug0 = pynof.compute_energy(mol,p)
~~~

If everything worked, the job may be executed by
~~~
python -u test_energy.py
~~~

*Note.* The first run may be slightly slow due to jit precompilation.

# :construction: For development

First, clone PyNOF from github and change to the project directory
~~~
git clone https://github.com/felipelewyee/PyNOF.git
cd PyNOF
~~~

In the PyNOF folder, execute the following code
~~~
conda create -y -n pynof
conda activate pynof
conda install -y psi4 -c conda-forge/label/libint_dev -c conda-forge
conda install -y -c conda-forge cupy # Optional
pip install build
python -m build && cd dist && pip install pynof*.whl && cd ..
~~~

# :sunglasses: Authors

The PyNOF code has been built by Juan Felipe Huan Lew Yee, Lizeth Franco Nolasco, Iván Alejandro Rivera and Roberto Rojas Hernández under supervision of Jorge Martín del Campo Ramírez and Mario Piris. We also acknowledge contributions from Rodrigo Vargas Hernández and José M. Mercero.
