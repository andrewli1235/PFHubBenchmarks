#
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from pfbase_ali_mod import *
from ufl import split, dx, ds, inner, grad, variable, diff

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
Lx = 25.
Ly = 1.
Nx = 250
Ny = 10
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE]
ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

ux, uy = df.split(w)
ux = df.variable(ux)
uy = df.variable(uy)


#dux, duy = df.split(dw)
#dux = df.variable(dux)
#duy = df.variable(duy)

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

E = Constant(1e5)
nu = Constant(0.3)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

#loading condition
rho_g = 1e-3
f = df.Constant((0, -rho_g))


def sigma(vx, vy, mu_0 = mu, lmbda_0 = lmbda):
    eps_xx, eps_xy, eps_yy = eps(vx, vy)
    
    
    sigma_xx = lmbda_0*(eps_xx+eps_yy) + 2*mu_0*eps_xx
    sigma_xy = 2*mu_0*eps_xy                # sigma_yx = sigma_xy
    sigma_yy = lmbda_0*(eps_xx+eps_yy) + 2*mu_0*eps_yy
    
    return sigma_xx, sigma_xy, sigma_yy

eps_xx, eps_xy, eps_yy = eps(ux, uy)
sigma_xx, sigma_xy, sigma_yy = sigma(ux, uy, mu, lmbda)
#sigma_xx_x, sigma_xx_y = df.grad(sigma_xx)
#sigma_xy_x, sigma_xy_y = df.grad(sigma_xy)
#sigma_yy_x, sigma_yy_y = df.grad(sigma_yy)


a_x = w_[0]*(sigma_xx*eps_xx+sigma_xy*eps_xy)*dx
l_x = 0

a_y =  w_[1]*(sigma_xy*eps_xy+sigma_yy*eps_yy)*dx

l_y = w_[1]*df.Constant(-rho_g)*dx

def left(x, on_boundary):
    return near(x[0], 0.)

bc = DirichletBC(W, Constant((0.,0.)), left)

def left(x, on_boundary):
    return near(x[0], 0.)

bc = DirichletBC(W, Constant((0.,0.)), left)


def F_ux_weak_form(vx_, vy_, vx, vy, fx, mu_0 = mu, lmbda_0 = lmbda):
    eps_xx, eps_xy, eps_yy = eps(vx_, vy_)
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, mu_0, lmbda_0)
    lhs = (sigma_xx*eps_xx+sigma_xy*eps_xy)*dx
    rhs = vx_*fx*dx

    F = lhs - rhs

    return F

def F_uy_weak_form(vx_, vy_, vx, vy, fy, mu_0 = mu, lmbda_0 = lmbda):
    eps_xx, eps_xy, eps_yy = eps(vx_, vy_)
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, mu_0, lmbda_0)

    lhs =  (sigma_xy*eps_xy+sigma_yy*eps_yy)*dx
    rhs = vy_*fy*dx

    F = lhs - rhs

    return F

def F_u_weak_form_tot(vx_, vy_, vx, vy, fx, fy, mu_0 = mu, lmbda_0 = lmbda):
    eps_xx, eps_xy, eps_yy = eps(vx_, vy_)
    sigma_xx, sigma_xy, sigma_yy = sigma(vx, vy, mu_0, lmbda_0)

    lhs = (sigma_xx*eps_xx+sigma_xy*eps_xy+sigma_xy*eps_xy+sigma_yy*eps_yy)*dx
    rhs = (vx_*fx+vy_*fy)*dx

    F = lhs  - rhs

    return F

#F_mx = F_ux_weak_form(w_[0], w_[1], w[0], w[1], f[0])
#F_my = F_ux_weak_form(w_[0], w_[1], w[0], w[1], f[1])
#F = F_mx + F_my

F = F_u_weak_form_tot(w_[0], w_[1], w[0], w[1], f[0], f[1])

Wux, Wuy = W.split()

tol = 1E-12
def boundary_left(x, on_boundary):
    return on_boundary and df.near(x[0], 0, tol)

bc_ux_left  = df.DirichletBC(Wux, df.Constant(0.0), boundary_left)
bc_uy_left  = df.DirichletBC(Wuy, df.Constant(0.0), boundary_left)



bcs = [bc_ux_left, bc_uy_left]  # no-flux on top, bottom boundary

###############
J = df.derivative(F, w, dw)


###################################
# Nonlinear solver setup
###################################
#df.set_log_level(df.LogLevel.ERROR)

problem = df.NonlinearVariationalProblem(F, w, bcs, J)
solver  = df.NonlinearVariationalSolver(problem)

#solver.parameters['nonlinear_solver'] = 'newton'
#nlparams  = solver.parameters['newton_solver']

solver.parameters['nonlinear_solver'] = 'snes'
nlparams  = solver.parameters['snes_solver']



nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 30
