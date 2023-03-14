# PFHubBenchmarks
PFHub benchmark problems implemented with FEniCS. Problem details https://pages.nist.gov/pfhub/

# Installation and Running

Install FEniCS using Miniconda.
```bash
conda create -n PF-env
conda activate PF-env
conda install -c conda-forge fenics dolfin-adjoint matplotlib
```
Clone the code

```bash
git clone git@github.com:vpuri3/PFHubBenchmarks.git
```

Run the code in parallel

```bash
cd PFHubBenchmarks/
mpirun -np 8 python dolfin/bench<1,2,3,6>.py
```
or in serial

```bash
cd PFHubBenchmarks/
python dolfin/bench<1,2,3,6>.py
```
