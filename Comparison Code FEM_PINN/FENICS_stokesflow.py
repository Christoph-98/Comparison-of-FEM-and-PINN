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
         ------>>>>>> u = 1.0 m/s   |      *                             *            |    
                      v = 0         |                             *                   |     bc from weak formulation          
                                    |                     *                           |
                                    |           *                *         *          |
                                0.4 |                                                 |
                                    |                                                 |
                                    |      *            *             *               |
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

import fenics as fe
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
import numpy.ma as ma 


N_POINTS_P_AXIS = 32
outfile_vel = "/mnt/c/Users/xxxxxxxxxxxxx/u_vel_fem_flow_test.txt"
outfile_pres = "/mnt/c/Users/xxxxxxxxxxxx/u_pres_fem_flow_test.txt"

KINEMATIC_VISCOSITY = 1   # -> μ = Viscosity
epsilon = 0.001

def main():
    start_timer = time.time()  # Starts the timer

    mesh = fe.UnitSquareMesh(N_POINTS_P_AXIS, N_POINTS_P_AXIS)

    # Function spaces: 
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

    # Boundary Conditions
    # The expression is used to mimic the parabolic flow which is expected here and also to not have the bc of 1 and 0 too close together
    inlet_velocity = fe.Expression(("4*x[1] * (1.0 - x[1])", "0.0"), degree=2)    
    #inlet_velocity = fe.Constant((1.0, 0.0))  
    zero_velocity = fe.Constant((0.0, 0.0))

    def top_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 1.0)

    def bottom_boundary(x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

    def left_boundary(x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

    # Fixing the presure so we dont get a singular system
    def fixed_pressure(x, on_boundary):
        return fe.near(x[0], 0.0) and fe.near(x[1], 0.0)
    
    bcs = [
        fe.DirichletBC(W.sub(0), inlet_velocity, left_boundary),
        fe.DirichletBC(W.sub(0), zero_velocity, top_boundary),
        fe.DirichletBC(W.sub(0), zero_velocity, bottom_boundary),
        fe.DirichletBC(W.sub(1), fe.Constant(0.0), fixed_pressure, method='pointwise')
    ]
   
    # Define the function f=0
    f = fe.Constant((0.0, 0.0))

    # Weak form of the steady Stokes equations
    a = (
        KINEMATIC_VISCOSITY * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx
        - fe.div(v) * p * fe.dx
        - q * fe.div(u) * fe.dx
    )
    # pressure stabilisation term
    a += epsilon * p * q * fe.dx

    L = fe.inner(f, v) * fe.dx  # way faster with fe.dot instead of fe.inner

    # Assemble and solve
    w = fe.Function(W)
    fe.solve(a == L, w, bcs, solver_parameters={"linear_solver": "lu"})

    # Split solution
    u_sol, p_sol = w.split()
    u_sol_out = u_sol.vector().get_local()
    
    # convergence tracking
    velocity_norm = fe.norm(u_sol, 'L2')
    print(f"L2 norm of velocity solution: {velocity_norm:.6f}")


    # time tracking
    end_timer = time.time()  # Ends the timer
    print(f"Simulation run time: {end_timer - start_timer:.3f} seconds")

    ux, uy = u_sol.split()

    plt.figure(figsize=(8, 6))
    p2 = fe.plot(p_sol, title="Pressure Field", cmap='cool')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.colorbar(p2)
    
    out_dir = Path("results_stokesflow")
    out_dir.mkdir(parents=True, exist_ok=True)   # creates it if it doesn't exist

    # Save the velocity and pressure DOF vectors
    np.savetxt(out_dir / "fem_sol_128.txt", u_sol_out, fmt="%.7f")

    # --- regular sampling grid -------------------------------------------
    n_grid = 100
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(x, y)

    # --- evaluate velocity on the grid ------------------------------------
    U = np.empty_like(X) * np.nan
    V = np.empty_like(Y) * np.nan

    for i in range(n_grid):
        for j in range(n_grid):
            pt = fe.Point(X[i, j], Y[i, j])
            U[i, j], V[i, j] = u_sol(pt)

    # --- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.streamplot(x, y, U, V, density=1.5, color=(0.6, 0.6, 0.6))
    velocity_plot = fe.plot(ux, cmap='jet')
    cbar = fig.colorbar(velocity_plot)    # attach colour‑bar to the line set
    cbar.set_label("Speed (|u|)")

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
    
    np.savetxt(outfile_vel,u_vals, fmt="%.7f", comments="")
    np.savetxt(outfile_pres,p_vals, fmt="%.7f", comments="")

if __name__ == "__main__":
    main()

