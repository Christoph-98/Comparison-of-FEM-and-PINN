
# FEniCs PINN AG
# linearized Stokes
#   Momntum:            ∇p + 𝜈∇^2u = f
#   Incompressibility:       div u = 0
#   Boundary condition:          u = 0

# u: Velocity 
# p: Pressure 
# 𝜈: Kinematic Viscosity 
# f: Extrnal Force (=0)

# v is the test function to the function of space of u
# q is the test function to the function of space of p

# Weak form: a(u,v) + b(v,p) = <f,v>
#                     b(u,q) = 0

"""

                        


                                                    u = 0
                                                    v = 0

                                1 +-------------------------------------------------+
                                    |                                                 |
                                    |             *                      *            |
                                    |          *           *    *    *                |
                                0.8 |                                                 |
                                    |                                 *               |
                                    |     *       *                                   |
                                    |                      *     *                    |
                                0.6 |                                            *    |
         ------>>>>>> u = 1.0 m/s   |      *            @@@@@@@@@        *            |    
                      v = 0         |              @@@@@@@@@@@@@@@@@@@                |     bc from weak formulation          
                                    |            @@@@@@@@@@@@@@@@@@@@@@@              |
                                    |           @@@@@@@@@@@@@@@@@@@@@@@@@       *     |
                                0.4 |            @@@@@@@@@@@@@@@@@@@@@@@              |
                                    |              @@@@@@@@@@@@@@@@@@@                |
                                    |      *            @@@@@@@@@             *       |
                                    |           *                             *       |
                                0.2 |                       *           *             |
                                    |                               *                 |
                                    |  *          *      *                 *       *  |
                                    |                            *                    |
                                0 +-------------------------------------------------+
                                    0        0.2       0.4       0.6       0.8        1

                                                            u = 0
                                                            v = 0


* Velocity components have zero initial condition.
* Homogeneous Dirichlet Boundary Conditions everywhere except for horizontal
  velocity at top. It is driven by an external flow.

-----

Denote by:

u   : The velocity field (from the previous iteration)
u*  : The velocity field after a tentative momentum step
u** : The velocity field after incompressibility correction (=after projection)

v   : The test function to the function space of u
q   : The test function to the function space of p


In order to guarantee stability of the spatial discretization, we will
use so-called Taylor-Hood elements -> The order of Ansatz/Shape functions
for the pressure function space has to be one order smaller than the 
velocity function space.

-----

Note on stability:

We use an explicit treatment of the convection/advection. Hence, choose
the time step length with care.
"""
import numpy as np
import fenics as fe
from fenics import *
import matplotlib.pyplot as plt
import mshr
import time
from pathlib import Path
import numpy.ma as ma 
from numpy.random import default_rng
rng = default_rng(seed=42)


N_POINTS_P_AXIS = 64
#outfile_vel = "/mnt/c/Users/xxxxxxxxxxxxxxx/u_vel_fem_object_test.txt"
#outfile_pres = "/mnt/c/Users/xxxxxxxxxxxxxx/u_pres_fem_object_test.txt"

KINEMATIC_VISCOSITY = 1
epsilon = 0.001

# Define the U-boat shape
center = fe.Point(1.0, 1.0)  # New center for the larger domain
a1 = 0.2
b1 = 0.1  

def main():
    start_timer = time.time()

    # Create domain: square [0,2] x [0,2] with elliptical hole
    square = mshr.Rectangle(Point(0.0, 0.0), Point(2.0, 2.0))
    ellipse = mshr.Ellipse(center, a1, b1)
    domain = square - ellipse
    mesh = mshr.generate_mesh(domain, N_POINTS_P_AXIS)

    print("Number of vertices:", mesh.num_vertices())

    # Function space
    V = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, fe.MixedElement([V, Q]))

    dof_coords = W.tabulate_dof_coordinates().reshape(-1, mesh.geometry().dim())
    print(f'this is len of dof coords',len(dof_coords))

    # Separate x and y
    x = dof_coords[:, 0]
    y = dof_coords[:, 1]

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, s=10, c='blue')
    plt.title("FEM DOF Coordinates (P2-P1 Mixed Element)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


    print(W.dim())          
    print(W.sub(0).dim())   
    print(W.sub(1).dim())   

    # Trial and test functions
    (u, p) = fe.TrialFunctions(W)
    (v, q) = fe.TestFunctions(W)

    # Inlet BC
    #inlet_velocity = fe.Expression(("2*x[1]*(2.0 - x[1])", "0.0"), degree=2)
    #inlet_velocity = fe.Constant((1.0, 0.0))


    class NoisyParabolicInlet(UserExpression):
        def eval(self, value, x):
            base = 2 * x[1] * (2.0 - x[1])  # original parabola
            noise = 0.0 * rng.normal()      # adjust magnitude as needed
            value[0] = base + noise
            value[1] = 0.0

        def value_shape(self):
            return (2,)

    inlet_velocity = NoisyParabolicInlet(degree=2)


    zero_velocity = fe.Constant((0.0, 0.0))

    def top_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 2.0)

    def bottom_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

    def left_boundary(x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

    def fixed_pressure(x, on_boundary):
        return fe.near(x[0], 0.0) and fe.near(x[1], 0.0)

    bcs = [
        fe.DirichletBC(W.sub(0), inlet_velocity, left_boundary),
        fe.DirichletBC(W.sub(0), zero_velocity, top_boundary),
        fe.DirichletBC(W.sub(0), zero_velocity, bottom_boundary),
        fe.DirichletBC(W.sub(0), zero_velocity,
                       "on_boundary && x[0]>0.8 && x[0]<1.2 && x[1]>0.9 && x[1]<1.1"),
        fe.DirichletBC(W.sub(1), fe.Constant(0.0), fixed_pressure, method='pointwise')
    ]
   
    f = fe.Constant((0.0, 0.0))

    # Weak form
    a = (
        KINEMATIC_VISCOSITY * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        - fe.div(v) * p * fe.dx
        - q * fe.div(u) * fe.dx
    )
    a += epsilon * p * q * fe.dx
    L = fe.dot(f, v) * fe.dx

    # Solve
    w = fe.Function(W)
    fe.solve(a == L, w, bcs, solver_parameters={"linear_solver": "lu"})

    u_sol, p_sol = w.split()
    u_sol_out = u_sol.vector().get_local()
    p_sol_out = p_sol.vector().get_local()
    print(len(u_sol_out))
    print(len(p_sol_out))

    # Ellipse outline
    theta = np.linspace(0, 2 * np.pi, 200)
    x_ellipse = center.x() + a1 * np.cos(theta)
    y_ellipse = center.y() + b1 * np.sin(theta)

    velocity_norm = fe.norm(u_sol, 'L2')
    print(f"L2 norm of velocity solution: {velocity_norm:.6f}")

    end_timer = time.time()
    print(f"Simulation run time: {end_timer - start_timer:.3f} seconds")
    
    ux, uy = u_sol.split()

    # Plot Velocity and Pressure
    #plt.figure(figsize=(6, 6))

    # Pressure field
    # pressure_plot = fe.plot(p_sol)
    # plt.plot(x_ellipse, y_ellipse, 'k-', linewidth=1)
    # plt.title("Pressure Field")
    # plt.colorbar(pressure_plot)
    # plt.tight_layout()

    # Sampling grid
    n_grid = 100
    x = np.linspace(0.0, 2.0, n_grid)
    y = np.linspace(0.0, 2.0, n_grid)
    y_vals = np.linspace(0.0, 2.0, n_grid)
    X, Y = np.meshgrid(x, y)
    u_vals = []

    
    for yy in y_vals:
        point = fe.Point(0.0, yy)
        val = inlet_velocity(point)
        u_vals.append(val[0])  # horizontal velocity (u_x)

    U = np.empty_like(X) * np.nan
    V = np.empty_like(Y) * np.nan

    tree = mesh.bounding_box_tree()

    for i in range(n_grid):
        for j in range(n_grid):
            pt = Point(X[i, j], Y[i, j])
            if tree.compute_first_entity_collision(pt) < mesh.num_cells():
                U[i, j], V[i, j] = u_sol(pt)

    fig, ax = plt.subplots(figsize=(8, 6))
    p2 = fe.plot(p_sol, title="Pressure Field", cmap='cool')
    # Obstacle outline
    ax.plot(center.x() + a1 * np.cos(theta),
            center.y() + b1 * np.sin(theta),
            "k", lw=1)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(p2)

    fig, ax = plt.subplots(figsize=(8, 6))
    # 1. Plot ux with custom colormap
    plt.plot(u_vals, y_vals, label="Inlet velocity $u_x(y)$",color='black')
    ax.streamplot(x, y, U, V, density=1.5, color=(0.6, 0.6, 0.6))
    velocity_plot = fe.plot(ux, cmap="jet") 
    cbar1 = fig.colorbar(velocity_plot, ax=ax, orientation='vertical')
    cbar1.set_label("x-velocity (uₓ)")

    # Obstacle outline
    ax.plot(center.x() + a1 * np.cos(theta),
            center.y() + b1 * np.sin(theta),
            "k", lw=1)

    ax.set_title("Velocity Field (uₓ)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # Sample u_sol on an evenly‑spaced grid 
    nx = ny = 64                         
    sample_mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(2.0, 2.0), nx, ny)

    # Collect the vertex coordinates 
    coords = sample_mesh.coordinates()   

    # Prepare an array for [u,v] at each point; default to NaN
    uv_vals = np.full((coords.shape[0], 2), np.nan)

    # Fast collision test so we don’t evaluate inside the obstacle
    tree = mesh.bounding_box_tree()

    for i, (x_i, y_i) in enumerate(coords):
        pt = fe.Point(x_i, y_i)
        if tree.compute_first_entity_collision(pt) < mesh.num_cells():
            # Point is inside the fluid domain – evaluate velocity there
            uv_vals[i, :] = u_sol(pt)

    # Replace NaNs with 0.0 in the u-component
    u_vals = np.nan_to_num(uv_vals[:, 0], nan=0.0)

    # Prepare an array for pressure at each point; default to NaN
    p_vals = np.full(coords.shape[0], np.nan)

    for i, (x_i, y_i) in enumerate(coords):
        pt = fe.Point(x_i, y_i)
        if tree.compute_first_entity_collision(pt) < mesh.num_cells():
            # Point is inside the fluid domain – evaluate pressure there
            p_vals[i] = p_sol(pt)

    # Replace NaNs with 0.0 if desired
    p_vals = np.nan_to_num(p_vals, nan=0.0)

    # Save to a text file
    # np.savetxt(outfile_vel,u_vals, fmt="%.7f", comments="")
    # np.savetxt(outfile_pres,p_vals, fmt="%.7f", comments="")

if __name__ == "__main__":
    main()


