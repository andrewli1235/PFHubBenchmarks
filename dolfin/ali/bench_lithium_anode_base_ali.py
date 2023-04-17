#
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

from pfbase_ali_mod import *
from ufl import split, dx, ds, inner, grad, variable, diff, ln

save_solution = True



##### weak forms we are trying #######

#for equation which couples a conserved and non-conserved time evolution together
def mixed_Li_weak_form(theta, xi, theta_, theta0, xi0, dt, D_eff, Omega_li, Omega_v, RT):

    # """

    # """

    # Li occupancy - theta
    Omega_diff = Omega_Li - Omega_v
    
    Fc_lhs =  theta_ * (h(xi)*(theta - theta0) / dt + theta*dh(xi) * (xi-xi0)/dt) * dx
    Fc_rhs = -inner(grad(theta_), 0.5 * D_eff * h(xi) * grad(theta)/RT) * dx
    Fc_rhs += inner(grad(theta_), D_eff * h(xi) * theta* Omega_diff * grad(theta)/RT) * dx
    
    F_c = Fc_lhs - Fc_rhs


    return F_c


def lattice_site_weak_form(theta, xi, xi_, theta_eq, xi0, dt, w, L, kappa, Omega_v, Omega_L, RT, dg):

    # """
    
    # """

    lhs  = (1/dt) * xi_ * (xi - xi0) * dx 
    rhs  = -L * xi_ *dh(xi)* RT * (1/Omega_L) * df.ln((1-theta)/(1-theta_eq)) *dx
    rhs += -L* xi_ * w * dg * dx #+ kappa * inner(grad(xi_), grad(xi)) * dx

    F_L = lhs - rhs

    return F_L


class InitialConditionsBench_void(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Lx = args[0]
        self.Ly = args[1]
        self.R = args[2]


    def eval(self, values, x):
        # indices
        # c, phi, x displacement, y displacement, ksi 
        # 0,  1,   2,              3,              4,
        

        values[0] = 0.95
        values[1] = 0.0
        values[2] = 0.0
        values[3] = 0.0
        
        r = np.sqrt((x[0]-0.5*self.Lx)**2 + (x[1]-0.5*self.Ly)**2)
        
        if r < self.R:
            values[4] = 0.0
        else:
            values[4] = 1.0
  

    def value_shape(self):
        return (5,)


###################################
# Optimization options for the finite element form compiler
###################################
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
df.parameters["form_compiler"]["quadrature_degree"] = 3

###################################
# Create or read mesh
###################################
Lx = Ly = 100.0      # 20 grids will act as the initial void location
Nx = Ny = 100        
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

# parameters

D_eff = 7.5e-13      # Effective diffusivity in Li metal m^2/s
E_Li = 4.9           # Young's modulus of Li metal in GPa
nu_Li = -0.38        # Poisson's ratio of Li metal
nu_LLZO = 0.257      # Poisson's ratio for LLZO
L = 1e-9             # Interface kinetics coefficient/AC equation kinetics
w_double = 3.5e6            # double well potential for lattice site time evolution
kappa = 4.5e-7       # gradient energy coefficient
R = 8.314            # J/(mol K)
T = 298              # Temp in Kelvin
RT =R * T
Omega_Li = 13.1e-6   # Li molar volume m^3/mol
Omega_v = 6e-6       # Vacancy molar volume m^3/mol
Omega_L = 13.1e-6    # Average molar volume of lattice sites m^3/mol
sigma_Li = 1.1e7     # Li metal electrical conductivity [S/m]
sigma_LLZO = 5.5e-6  # LLZO ionic conductivity [S/m]
theta_eq = 0.95

# dummy poisson setup
k = 1.0
epsilon = 1.0

# FEM setup
P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE, PE, PE, PE] # c, phi, ux, uy, ksi

ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

###### Initial conditions #####
radius = 10.0

w0 = df.Function(W)
w_ic = InitialConditionsBench_void(Lx, Ly, radius, degree=2)
w0.interpolate(w_ic)

# Free energy functional
theta, phi, ux, uy, xi = df.split(w)
theta   = df.variable(theta)
phi = df.variable(phi)
ux = df.variable(ux)
uy = df.variable(uy)
xi = df.variable(xi)

boundary_markers = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1,0)

class Boundary_right(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)
    
class Boundary_top_bottom(SubDomain):
    tol = 1E-14
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0, tol) or near(x[1], Ly, tol))
    

b_r = Boundary_right()
b_t_b = Boundary_top_bottom()


b_r.mark(boundary_markers, 0)
b_t_b.mark(boundary_markers, 1)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

j_n_theta = 1.0 # right boundary condition


# BC
tol = 1E-12
def boundary_left(x, on_boundary):
    return on_boundary and df.near(x[0], 0, tol)

def boundary_right(x, on_boundary):
    return on_boundary and df.near(x[0], Lx, tol)

def boundary_down(x, on_boundary):
    return on_boundary and df.near(x[1], 0, tol)

def boundary_up(x, on_boundary):
    return on_boundary and df.near(x[1], Ly, tol)

phi_right = df.Expression(("sin(x[1]/7)"), degree=2)


Wtheta, Wphi, Wux, Wuy, Wxi = W.split()

bc_phi_left  = df.DirichletBC(Wphi, df.Constant(0.0), boundary_left)
bc_phi_right = df.DirichletBC(Wphi, phi_right, boundary_right)

bc_ux_left  = df.DirichletBC(Wux, df.Constant(0.0), boundary_left)
bc_ux_right = df.DirichletBC(Wux, df.Constant(0.0), boundary_right)

bc_uy_left  = df.DirichletBC(Wuy, df.Constant(0.0), boundary_left)
bc_uy_right = df.DirichletBC(Wuy, df.Constant(0.0), boundary_right)

bcs = [bc_phi_left, bc_phi_right, bc_ux_left, bc_ux_right, bc_uy_left, bc_uy_right]  # no-flux on top, bottom boundary

# this function returns the total strain tensor components
# vx = x component of the displacement vector, vy = y component of the displacement
'''
def eps(vx, vy):
    duxdx, duxdy = df.grad(vx)
    duydx, duydy = df.grad(vy)
    eps_xx = duxdx
    eps_xy = 0.5*(duxdy+duydx) # eps_yx = eps_xy
    eps_yy = duydy
    return eps_xx, eps_xy, eps_yy 

#plane stress case
def sigma(vx, vy, c):
    h_temp = h(c)
    modulus_mod = (1+0.1*h_temp)
    c_1111 = 250.0 * modulus_mod
    c_1122 = 150.0 * modulus_mod
    c_1212 = 100.0 * modulus_mod
    eps_chem = 0.005
    eps_xx, eps_xy, eps_yy = eps(vx, vy)
    
    #obtain elastic strain term
    eps_el_xx = eps_xx-eps_chem*h_temp
    eps_el_yy = eps_yy-eps_chem*h_temp
    eps_el_xy = eps_xy
    
    sigma_xx = c_1111*eps_el_xx+c_1122*eps_el_yy #subtract out chemical contribution
    sigma_xy = c_1212*eps_el_xy*2                # sigma_yx = sigma_xy
    sigma_yy = c_1111*eps_el_yy+c_1122*eps_el_xx
    
    return sigma_xx, sigma_xy, sigma_yy


#elastic free energy contribution
def f_el(vx, vy, c):
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, c)
    eps_xx, eps_xy, eps_yy = eps(vx, vy)
    return 0.5*(eps_xx*sigma_xx+eps_xy*sigma_xy*2+eps_yy*sigma_yy)


# mechanical equilibrium constitutive equation
def F_u_weak_form_tot(vx_, vy_, vx, vy, c, fx, fy):
    eps_xx, eps_xy, eps_yy = eps(vx_, vy_)
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, c)

    lhs = (sigma_xx*eps_xx+sigma_xy*eps_xy+sigma_xy*eps_xy+sigma_yy*eps_yy)*dx
    rhs = (vx_*fx+vy_*fy)*dx

    F = lhs  - rhs

    return F
'''

#simplified elastic free energy contribution
def f_el(c, p):
    
    return 1


F_theta = mixed_Li_weak_form(theta, xi, w_[0], w0[0], w0[4], dt, D_eff, Omega_Li, Omega_v, RT)

F_theta += w_[0] * j_n_theta * h(xi) * ds(0)
F_theta += w_[0] *  0.0 * ds(1)


Fp = poisson_weak_form(w[1], w_[1], -k * w[0] / epsilon, df.Constant(1.0))
F_mx = euler_bwd_weak_form(w[2], w_[2], df.Constant(0.0), dt, df.Constant(0.0))
F_my = euler_bwd_weak_form(w[3], w_[3], df.Constant(0.0), dt, df.Constant(0.0))
F_1 =  Fp + F_mx + F_my

dg = 2*xi*(1-2*xi**2) 

F_xi = lattice_site_weak_form(theta, w[4], w_[4], theta_eq, w0[4], dt, w_double, L, kappa, Omega_v, Omega_L, RT, dg)
F_xi += w_[4] * 0.0 * ds(0)
F_xi += w_[4] * 0.0 * ds(1)
F = F_1 + F_xi


###############
J = df.derivative(F, w, dw)

###################################
# Nonlinear solver setup
###################################
df.set_log_level(df.LogLevel.ERROR)

problem = df.NonlinearVariationalProblem(F, w, bcs, J)
solver  = df.NonlinearVariationalSolver(problem)

#solver.parameters['nonlinear_solver'] = 'newton'
#nlparams  = solver.parameters['newton_solver']

solver.parameters['nonlinear_solver'] = 'snes'
nlparams  = solver.parameters['snes_solver']

nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 10

#
# bactracig (bt) diverges with only Laplace eqn
#nlparams['line_search'] = 'bt'      # WORKS (7s) for np=32, T=3.0
nlparams['line_search'] = 'cp'       # (8s) #
#nlparams['line_search'] = 'basic'   # (7s)
#nlparams['line_search'] = 'nleqerr' # (15s)
#nlparams['line_search'] = 'l2'      # FAILING

# 
nlparams['linear_solver'] = 'gmres'
#nlparams['preconditioner'] = 'sor'

#nlparams['linear_solver'] = 'gmres'
#nlparams['linear_solver'] = 'bicgstab'
#nlparams['linear_solver'] = 'minres'

#nlparams['preconditioner'] = 'none'
#nlparams['preconditioner'] = 'sor'
#nlparams['preconditioner'] = 'petsc_amg'
nlparams['preconditioner'] = 'hypre_amg'

nlparams['krylov_solver']['maximum_iterations'] = 5000
#nlparams['krylov_solver']['monitor_convergence'] = True


###################################
# analysis setup
###################################
if save_solution:
    filename = "results/bench_ali_ac/conc"+".h5"
    if os.path.isfile(filename):
        os.remove(filename)
    outfile = HDF5File(MPI.comm_world, filename, "w")
    outfile.write(mesh, "mesh")

def total_solute(c):
    return df.assemble(c * dx)

def total_free_energy(f_chem, f_elec, f_m, kappa):
    E = df.assemble((
        f_chem +
        f_elec +
        f_m +
        kappa / 2.0 * inner(grad(c), grad(c))
        )*dx)

    return E

###################################
# time integration
###################################

# Ensure everything is reset
t = df.Constant(0.0)
tprev = 0.0
w.interpolate(w_ic)
w0.interpolate(w_ic)

benchmark_output = []
end_time = df.Constant(3) # 400.0
iteration_count = 0
dt_min = 1e-4
dt.assign(1e-2)
t1 = time.time()

while float(t) < float(end_time) + df.DOLFIN_EPS:

    tprev = float(t)

    iteration_count += 1
    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
    else:
        pass

    # set IC
    w0.assign(w)

    # solve
    t.assign(tprev + float(dt))
    niters, converged = solver.solve()

    while not converged:
        #if float(dt) < dt_min + 1E-8:
        #    if df.MPI.rank(mesh.mpi_comm()) == 0:
        #        print("dt too small. exiting.")
        #    postprocess()
        #    exit()

        dt.assign(max(0.5*float(dt), dt_min))
        t.assign(tprev + float(dt))
        w.assign(w0)

        if df.MPI.rank(mesh.mpi_comm()) == 0:
            print(f'REPEATING Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
        niters, converged = solver.solve()

    # Simple rule for adaptive timestepping
    if (niters < 5):
        dt.assign(2*float(dt))
    else:
        dt.assign(max(0.5*float(dt), dt_min))

    ############
    # Analysis
    ############
    c, phi, ux, uy, ksi = w.split()

    if save_solution:
        outfile.write(c , "c" , float(t))

    F_total = 1 #total_free_energy(f_chem, f_elec, f_m, kappa)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

t2 = time.time()
spent_time = t2 - t1
if df.MPI.rank(mesh.mpi_comm()) == 0:
    print(f'Time spent is {spent_time}')
else:
    pass

###################################
# post process
###################################
if df.MPI.rank(mesh.mpi_comm()) == 0:
    np.savetxt('results/bench_ali_ac' + '_out.csv',
            np.array(benchmark_output),
            fmt='%1.10f',
            header="time,total_free_energy,total_solute",
            delimiter=',',
            comments=''
            )
else:
    pass
