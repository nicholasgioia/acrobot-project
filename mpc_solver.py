import numpy as np
import casadi as ca
from config import Config

cfg = Config()

def ltv_mpc(A_seq, B_seq, Q, R, Qf, x_err_0, u_ref_seq, x_ref_seq, constraints):
    """
    Solves a Linear Time-Varying MPC problem on the error dynamics using CasADi.
    
    Instead of optimizing the full state, this solver optimizes the 'deviation'
    from a reference trajectory, which is more computationally efficient for
    tracking tasks.
 
    Parameters:
        A_seq: (nx, nx, N) Sequence of linearized state matrices along the horizon.
        B_seq: (nx, nu, N) Sequence of linearized input matrices along the horizon.
        Q, R, Qf: Cost weighting matrices (Stage State, Input, and Terminal State).
        x_err_0: Current state error (x_actual - x_reference).
        u_ref_seq: Reference inputs over the horizon.
        x_ref_seq: Reference states over the horizon.
        constraints: Dictionary containing physical limits (min/max torque and angles).
        
    Returns:
        delta_u_opt: The optimal torque increment for the current time step.
        feasible: Boolean indicating if the optimizer found a valid solution.
    """
    
    # Dimensions
    nx = A_seq.shape[0]
    nu = B_seq.shape[1]
    N = A_seq.shape[2] # Prediction Horizon
    
    # Initialize CasADi optimization object
    opti = ca.Opti()
    
    # Decision Variables (deviations)
    # dX: State error sequence [dx_0, ..., dx_N]
    X = opti.variable(nx, N + 1)
    # dU: Input error sequence [du_0, ..., du_N-1]
    U = opti.variable(nu, N)
    
    cost = 0
      
    # Constraints:
    #     delta_x_{k+1} = A_k * delta_x_k + B_k * delta_u_k
    #     U_min <= u_ref + delta_u <= U_max
    #     X_min <= x_ref + delta_x <= X_max
        
    # Constraints unpacking
    u_min = constraints['u_min']
    u_max = constraints['u_max']
    # If state constraints exist
    x_min = constraints.get('x_min', -np.inf * np.ones(nx))
    x_max = constraints.get('x_max', np.inf * np.ones(nx))

    # Initial Constraint
    opti.subject_to(X[:, 0] == x_err_0)
    
    # Horizon Loop
    for k in range(N):
        # Dynamics Constraint (LTV) 
        # dx_{k+1} = A_k * dx_k + B_k * du_k
        # Note: A_seq[:,:,k] is numpy, we convert to CasADi MX/DM automatically by operation
        opti.subject_to(X[:, k+1] == A_seq[:,:,k] @ X[:, k] + B_seq[:,:,k] @ U[:, k])
        
        if cfg.TASK_4_CONSTRAINTS_ON:
            # Input Constraints
            # u_real = u_ref + du
            # u_min <= u_ref_seq[:, k] + dU[:, k] <= u_max
            u_ref_k = u_ref_seq[:, k]
            opti.subject_to(opti.bounded(u_min, u_ref_k + U[:, k], u_max))
            
            # State Constraints
            # x_real = x_ref + dx
            x_ref_k = x_ref_seq[:, k] # Excludes N+1 for now, check loop index
            # To avoid infeasibility due to noise/perturbations at k=0, 
            # sometimes strict constraints on x0 are relaxed or constraints start from k=1.
            if k > 0:
                opti.subject_to(opti.bounded(x_min, x_ref_k + X[:, k], x_max))
        
        # Cost Calculation
        # LQR Stage Cost: dx' Q dx + du' R du
        cost += ca.mtimes([X[:, k].T, Q, X[:, k]]) + ca.mtimes([U[:, k].T, R, U[:, k]])
        
    # Terminal Cost
    cost += ca.mtimes([X[:, N].T, Qf, X[:, N]])
    
    # Solve
    opti.minimize(cost)
    
    # Configure IPOPT (Interior Point Optimizer) settings
    opts = {
        'ipopt.print_level': 0,                 # Suppress solver output
        'print_time': 0,                        # Suppress timing info
        'ipopt.sb': 'yes',                      # Suppress IPOPT banner
        'ipopt.max_iter': 100,                  # Maximum solver iterations per time step
        'ipopt.warm_start_init_point': 'yes'    # Use previous solution as a starting guess for speed
    }
    opti.solver('ipopt', opts)
    
    try:
        sol = opti.solve()
        # Return first optimal input deviation
        return sol.value(U[:, 0]), True
    except RuntimeError:
        # Fallback if infeasible: return zero deviation
        print("MPC Infeasible/Failed")
       
        return np.zeros(nu), False