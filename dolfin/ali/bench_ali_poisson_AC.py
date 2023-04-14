#
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from pfbase_ali_mod import *
from ufl import split, dx, inner, grad, variable, diff

save_solution = True

###################################
# Optimization options for the finite element form compiler
###################################
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
df.parameters["form_compiler"]["quadrature_degree"] = 3

###################################
# Create or read mesh
###################################
Lx = Ly = 100.0
Nx = Ny = 100
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

# CH parameters
c_alpha = df.Constant(0.3)
c_beta = df.Constant(0.7)
kappa_c = df.Constant(2.0)
rho = df.Constant(5.0)
M = df.Constant(5.0)
k = df.Constant(0.09)



# AC parameters
kappa_eta = df.Constant(3.0)
ww = df.Constant(1.0)
alpha = df.Constant(5.0)
L = df.Constant(5.0)


NUM_ETA = 1

# FEM setup
P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE, PE, PE, PE]

for i in range(NUM_ETA):
    ME.append(PE)

ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

###### Initial conditions #####

##### INCOMPLETE #####
cc0 = 0.5
cc1 = 0.01

epsilon = df.Constant(90.0)
epsilon_eta = 0.1
psi = 1.5

w0 = df.Function(W)
w_ic = InitialConditionsBench_ali_AC(cc0, cc1, epsilon_eta, psi, degree=2)
w0.interpolate(w_ic)

# Free energy functional
c, _, phi, ux, uy, eta = df.split(w)
c   = df.variable(c)
phi = df.variable(phi)
ux = df.variable(ux)
uy = df.variable(uy)
eta = df.variable(eta)

# this function returns the total strain tensor components
# vx = x component of the displacement vector, vy = y component of the displacement
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

"""
def F_ux_weak_form(ux_, sigma_xx, sigma_xy):
    
    sigma_xx_x, sigma_xx_y = df.grad(sigma_xx)
    sigma_xy_x, sigma_xy_y = df.grad(sigma_xy)

    lhs =  ux_*(sigma_xx_x+sigma_xy_x)*dx
    rhs = 0

    F = lhs - rhs

    return F

def F_uy_weak_form(uy_, sigma_yy, sigma_xy):
    
    sigma_yy_x, sigma_yy_y = df.grad(sigma_yy)
    sigma_xy_x, sigma_xy_y = df.grad(sigma_xy)

    lhs =  uy_*(sigma_yy_y+sigma_xy_y)*dx
    rhs = 0

    F = lhs - rhs

    return F
"""

def F_ux_weak_form(ux_, vx, vy, c):
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, c)
    sigma_xx_x, sigma_xx_y = df.grad(sigma_xx)
    sigma_xy_x, sigma_xy_y = df.grad(sigma_xy)

    lhs =  ux_*(sigma_xx_x+sigma_xy_x)*dx
    rhs = 0

    F = lhs - rhs

    return F

def F_uy_weak_form(uy_, vx, vy, c):
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, c)
    sigma_yy_x, sigma_yy_y = df.grad(sigma_yy)
    sigma_xy_x, sigma_xy_y = df.grad(sigma_xy)

    lhs =  uy_*(sigma_yy_y+sigma_xy_y)*dx
    rhs = 0

    F = lhs - rhs

    return F

###### UNUSED ######

#f_chem = rho * (c - c_alpha)**2 * (c_beta - c)**2


# chemical free energy density for benchmark 4
a_0 = df.Constant(0)
a_1 = df.Constant(0)
a_2 = df.Constant(8.072789087)
a_3 = df.Constant(-81.24549382)
a_4 = df.Constant(408.0297321)
a_5 = df.Constant(-1244.129167)
a_6 = df.Constant(2444.046270)
a_7 = df.Constant(-3120.635139)
a_8 = df.Constant(2506.663551)
a_9 = df.Constant(-1151.003178)
a_10 = df.Constant(230.2006355)

#barrier height for benchmark 4
wall = 0.1

# CH term for benchmark 4
kappa = 0.29

f_chem = wall*(a_0+a_1*c+a_2*(c**2)+a_3*(c**3)+a_4*(c**4)
            +a_5*(c**5)+a_6*(c**6)+a_7*(c**7)+a_8*(c**8)
            +a_9*(c**9)+a_10*(c**10))

def double_well(u1, alpha):
    W = (u1**2 * (1 - u1)**2)

    return W

def hinterp(u1):
    return u1**3 * (6*u1**2 - 15*u1 + 10)

f_alpha = rho**2 * (c - c_alpha)**2
f_beta  = rho**2 * (c - c_beta)**2
f_chem  = (f_alpha * (1 - hinterp(eta)) +
           f_beta  * hinterp(eta) +
           ww * double_well(eta, alpha))

dfdc  = df.diff(f_chem, c)
dfdeta = df.diff(f_chem, eta)



f_chem = rho * (c - c_alpha)**2 * (c_beta - c)**2

f_elec = k * c * phi / 2.0

f_m = f_el(ux, uy, c)

dfdc = df.diff(f_chem, c) + k * phi + df.diff(f_m, c)



sigma_xx, sigma_xy, sigma_yy = sigma(w[3], w[4], c)

## weak form
Fc  = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa_c, dfdc)
Fp = poisson_weak_form(w[2], w_[2], -k * c / epsilon, df.Constant(1.0))
#Fp = euler_bwd_weak_form(w[2], w_[2], df.Constant(0.0), dt, df.Constant(0.0))
#F_mx = F_ux_weak_form(w_[3], w[3], w[4], w[0])
F_mx = euler_bwd_weak_form(w[3], w_[3], df.Constant(0.0), dt, df.Constant(0.0))
F_my = euler_bwd_weak_form(w[4], w_[4], df.Constant(0.0), dt, df.Constant(0.0))
Fe = allen_cahn_weak_form(w[5], w_[5], w0[5], dt, L, kappa_eta, dfdeta, df.Constant(0))



F= Fc + Fp + F_mx + F_my + Fe

# BC
tol = 1E-12
def boundary_left(x, on_boundary):
    return on_boundary and df.near(x[0], 0, tol)

def boundary_right(x, on_boundary):
    return on_boundary and df.near(x[0], Lx, tol)

phi_right = df.Expression(("sin(x[1]/7)"), degree=2)


_, _, Wphi, Wux, Wuy, Weta = W.split()
bc_phi_left  = df.DirichletBC(Wphi, df.Constant(0.0), boundary_left)
bc_phi_right = df.DirichletBC(Wphi, phi_right, boundary_right)

bc_ux_left  = df.DirichletBC(Wux, df.Constant(0.0), boundary_left)
bc_ux_right = df.DirichletBC(Wux, df.Constant(0.0), boundary_right)

bc_uy_left  = df.DirichletBC(Wuy, df.Constant(0.0), boundary_left)
bc_uy_right = df.DirichletBC(Wuy, df.Constant(0.0), boundary_right)



bcs = [bc_phi_left, bc_phi_right, bc_ux_left, bc_ux_right, bc_uy_left, bc_uy_right]  # no-flux on top, bottom boundary

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
nlparams['preconditioner'] = 'sor'

#nlparams['linear_solver'] = 'gmres'
#nlparams['linear_solver'] = 'bicgstab'
#nlparams['linear_solver'] = 'minres'

#nlparams['preconditioner'] = 'none'
#nlparams['preconditioner'] = 'sor'
#nlparams['preconditioner'] = 'petsc_amg'
#nlparams['preconditioner'] = 'hypre_amg'

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
end_time = df.Constant(1) # 400.0
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
    c, _, phi, ux, uy, eta = w.split()

    if save_solution:
        outfile.write(c , "c" , float(t))

    F_total = total_free_energy(f_chem, f_elec, f_m, kappa)
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
