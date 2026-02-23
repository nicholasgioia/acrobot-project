PROJECT: OPTIMAL CONTROL OF A PLANAR GYMNAST ROBOT (ACROBOT)
==============================================================================
 
1. OVERVIEW
------------------------------------------------------------------------------
This project implements an optimal control framework for a planar gymnast robot 
(Acrobot), modeled as a double pendulum actuated at the hip. 
The repository includes dynamics modeling, trajectory optimization via 
Regularized Newton's Method, and tracking using LQR and LTV-MPC.
 
2. REPOSITORY STRUCTURE
------------------------------------------------------------------------------
A. CORE & CONFIGURATION
   - main.py: Entry point executing Tasks 0-4 based on configuration.
   - config.py: Defines physical parameters (mass, length, inertia), 
     simulation settings (dt, horizon), and cost matrices.
 
B. DYNAMICS & MODELING
   - dynamics.py: Implements symbolic equations of motion using CasADi, 
     including Inertia, Coriolis, Gravity, and Friction terms. Provides RK4 
     integration and linearization.
   - equilibrium.py: Numerically computes equilibrium states and holding 
     torques using root-finding algorithms.
 
C. OPTIMIZATION & CONTROL
   - newtons_method.py: Implements the Regularized Newton's method for 
     iterative trajectory optimization.
   - LQR_affine.py: Solves the Finite-Horizon LQR problem for LTV systems 
     with affine terms, used for the backward pass.
   - mpc_solver.py: Implements Linear Time-Varying MPC using CasADi to solve 
     constrained optimization problems over a prediction horizon.
   - armijo.py: Implements Armijo backtracking line search for optimal 
     step-size selection.
   - cost.py: Defines stage and terminal cost functions (Quadratic) and 
     their gradients.
   - trajectory.py: Generates initial guess trajectories using Step inputs 
     or 5th-order Polynomials.
 
D. UTILITIES
   - animation.py: Renders robot dynamics and plots state evolution.
   - plot_task.py: Generates performance plots (states, inputs, descent).
 
3. REQUIREMENTS
------------------------------------------------------------------------------
   - numpy
   - scipy
   - matplotlib
   - casadi
   - control
 
4. EXECUTION
------------------------------------------------------------------------------
    1. Configure 'config.py' to adjust parameters or enable/disable tasks.
    2. Run the main script:
    $ python main.py
 
5. TASK SUMMARY
------------------------------------------------------------------------------
Task 0: Dynamics Validation
   Verifies discretized dynamics by maintaining a stationary equilibrium.
 
Task 1: Trajectory Optimization (Step Initialization)
   Computes an optimal transition between equilibria using Newton's method, 
   initialized with a step trajectory.
 
Task 2: Trajectory Optimization (Polynomial Initialization)
   Performs optimization using a smooth 5th-order polynomial reference.
 
Task 3: LQR Tracking
   Linearizes dynamics around the Task 2 optimal trajectory and computes 
   feedback gains to track the reference under perturbations.
 
Task 4: MPC Tracking
   Implements LTV-MPC to track the reference trajectory subject to input 
   and state constraints.
 
AUTHORS
------------------------------------------------------------------------------
Nicholas Gioia
Tommaso Scagliarini
Giorgio Soricetti