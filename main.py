from config import Config
from dynamics import AcrobotDynamics
from trajectory import TrajectoryGenerator
import cost as cst
import equilibrium as eq
import LQR_affine as lqr
import newtons_method as newton
import armijo as armijo
import plot_task as plotter
import numpy as np
import matplotlib.pyplot as plt
import animation as an
import mpc_solver as mpc

dyn = AcrobotDynamics()
cfg = Config()
traj_gen = TrajectoryGenerator(dyn)


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

nx = cfg.nx
nu = cfg.nu
T = cfg.N_steps
dt = cfg.dt


# Definition of task 0 function
def task_0():
    """
    TASK 0: EQUILIBRIUM VERIFICATION
    Ensures that the numerical equilibrium found by the solver is physically valid
    by checking if the system moves when starting from that state.
    """
    
    # TEST 1: Verify equilibrium point 
    print("\n==============================")
    print("Task 0: TEST Verify equilibrium point")
    print("==============================\n")

    x_A, u_A = eq.equilibrium_finding(cfg.TASK_0_THETA2_START_DEG)
    x_B, u_B = eq.equilibrium_finding(cfg.TASK_0_THETA2_END_DEG)

    # Set states and inputs to equilibrium values
    x_eq = x_A
    u_eq = u_A

    # Computing next state
    x_next = dyn.rk4_step(x_eq, u_eq)

    A, B = dyn.get_linearization_np(x_eq, u_eq)

    # Check if the state changes significantly
    diff = np.linalg.norm(x_next - x_eq)
    print(f"Initial state: {np.degrees(x_eq)}")
    print(f"Next state:    {np.degrees(x_next)}")
    
    if diff < 1e-9:
        print(f">> SUCCESS: The system stays in equilibrium. Diff: {np.degrees(diff)}")
        print(f"Shape of the linearization A matrix at equilibrium: {A.shape}\n")
        print(f"Shape of the linearization B matrix at equilibrium: {B.shape}\n")

    else:
        print(f">> WARNING: The system moved! Diff: {np.degrees(diff)}\n")


# Definition of task 1 function
def task_1(th2_A = cfg.TASK_1_THETA2_START_DEG, th2_B = cfg.TASK_1_THETA2_END_DEG):
    """
    TASK 1: TRAJECTORY GENERATION (STEP REFERENCE)
    Generates an optimized trajectory to move between two points using
    Newton's Method, starting from a discontinuous step reference.
    """

    print("\n==============================")
    print("Task 1 - Trajectory generation (I)")
    print("==============================\n")

    # Find equilibria for the two target theta2 angles
    x_A, u_A = eq.equilibrium_finding(th2_A)
    
    x_B, u_B = eq.equilibrium_finding(th2_B)

    print(f"Trajectory generation from theta2={th2_A} to theta2={th2_B}\n")

    # Generate Step Trajectory
    xx_step, uu_step = traj_gen.generate_step_trajectory(x_A, x_B, u_A, u_B)

    # Preallocate state and input trajectories for Newton's method
    xx = np.zeros((nx, T, cfg.TASK_1_MAX_ITERS+1))   
    uu = np.zeros((nu, T, cfg.TASK_1_MAX_ITERS+1))

    # Initial Guess (k = 0)
    # Fill all time steps with the first instant of state and input of the step trajectory, 
    # so that the initial guess is constant at the initial value of the reference trajectory, 
    # then they will be updated by Newton's method over the iterations
    for tt in range(T):
        xx[:,tt,0] = np.copy(xx_step[:,0])
        uu[:,tt,0] = np.copy(uu_step[:,0]) 
    x0=np.copy(xx_step[:,0])

    # Run Newton's method
    xx_opt, uu_opt, descent, J, kk = newton.newton_method(
        dyn, xx, uu, xx_step, uu_step, x0, cfg.TASK_1_MAX_ITERS, task_number=1,
        armijo_plot=cfg.TASK_1_PLOT_ARMIJO, armijo_plot_number=cfg.TASK_1_ARMIJO_PLOT_NUMBER
    )

    # Extraction of the final results
    xx_final = xx_opt[:,:,kk]
    uu_final = uu_opt[:,:,kk]

    # Plotting the results
    # Temporal axis creation
    time_axis = np.arange(T) * dt 

    if cfg.TASK_1_PLOT_RESULTS:
        plotter.plot_results_task1(time_axis, xx_step, uu_step, xx_opt, uu_opt, J, descent, kk)

    # Animating the results
    if cfg.TASK_1_ANIMATION:
        ani = an.animate_task_trajectory(time_axis, xx_final, xx_step, task_id=1)
        plt.tight_layout()
        plt.show()


# Definition of task 2 function
def task_2(th2_A = cfg.TASK_2_THETA2_START_DEG, th2_B = cfg.TASK_2_THETA2_END_DEG):
    """
    TASK 2: TRAJECTORY GENERATION (POLYNOMIAL REFERENCE)
    Generates a smooth 5th-order polynomial as the initial reference.
    This trajectory is used for tracking in Tasks 3 and 4.
    """

    print("\n==============================")
    print("Task 2 - Trajectory generation (II)")
    print("==============================\n")
    
    # Find equilibria for the two target theta2 angles
    x_A, u_A = eq.equilibrium_finding(th2_A)
    
    x_B, u_B = eq.equilibrium_finding(th2_B)

    print(f"Trajectory generation from theta2={th2_A} to theta2={th2_B}\n")

    # Generate Polynomial Trajectory
    xx_poly, uu_poly = traj_gen.generate_polynomial_trajectory(x_A, x_B, u_A, u_B)

    # Preallocate state and input trajectories for Newton's method
    xx = np.zeros((nx, T, cfg.TASK_2_MAX_ITERS+1))   
    uu = np.zeros((nu, T, cfg.TASK_2_MAX_ITERS+1))

    # Initial Guess (k = 0)
    # Fill all time steps with the first instant of state and input of the polynomial trajectory, 
    # so that the initial guess is constant at the initial value of the reference trajectory, 
    # then they will be updated by Newton's method over the iterations
    for tt in range(T):
        xx[:,tt,0] = np.copy(xx_poly[:,0])
        uu[:,tt,0] = np.copy(uu_poly[:,0]) 
    x0=np.copy(xx_poly[:,0])

    # Run Newton's method
    xx_opt, uu_opt, descent, J, kk = newton.newton_method(
        dyn, xx, uu, xx_poly, uu_poly, x0, cfg.TASK_2_MAX_ITERS, task_number=2,
        armijo_plot=cfg.TASK_2_PLOT_ARMIJO, armijo_plot_number=cfg.TASK_2_ARMIJO_PLOT_NUMBER
    )

    # Extraction of the final results
    xx_final = xx_opt[:,:,kk]
    uu_final = uu_opt[:,:,kk]
    uu_final[:,-1] = uu_final[:,-2] # Hold of the last valid value

    # Plotting the results
    # Temporal axis creation
    time_axis = np.arange(T) * dt 

    if cfg.TASK_2_PLOT_RESULTS:
        plotter.plot_results_task2(time_axis, xx_poly, uu_poly, xx_opt, uu_opt, J, descent, kk)

    # Animating the results
    if cfg.TASK_2_ANIMATION:
        ani = an.animate_task_trajectory(time_axis, xx_final, xx_poly, task_id=2)
        plt.tight_layout()
        plt.show()


    return xx_final, uu_final, x0


# Definition of task 3 function
Qt_LQR = cfg.Qt_task3
Rt_LQR = cfg.Rt_task3
QT_LQR = cfg.QT_task3

def task_3(xx_star, uu_star, x0):
    """
    TASK 3: TRAJECTORY TRACKING VIA LQR
    Applies Time-Varying LQR to track the reference trajectory from Task 2,
    handling initial state perturbations.
    """

    print("\n==============================")
    print("Task 3 - TRACKING VIA LQR ")
    print("==============================\n")

    # Initialize the algorithm
    AA = np.zeros((nx, nx, T))
    BB = np.zeros((nx, nu, T))
    QQ = np.zeros((nx, nx, T))
    RR = np.zeros((nu, nu, T))

    # Initialize terms that aren't used to zero but are needed for the function call of the affine LQR
    SS = np.zeros((nu, nx, T))  # Cross-term matrix (not used, set to zero)
    qq = np.zeros((nx, T))    # State affine term (not used, set to zero)
    rr = np.zeros((nu, T))    # Input affine term (not used, set to zero)
    qqf = np.zeros(nx)     # Terminal state affine term (not used, set to zero)
    
    # Linearization along the trajectory
    for t in range(T):
      dfx, dfu = dyn.get_linearization_np(xx_star[:, t], uu_star[:, t])
      AA[:, :, t] = dfx 
      BB[:, :, t] = dfu
      QQ[:, :, t] = Qt_LQR
      RR[:, :, t] = Rt_LQR

    # Set initial conditions with some perturbations
    x0_lqr = x0 + cfg.TASK_3_PERTURBATION

    # Solve the LQR problem
    KK_reg = lqr.ltv_LQR_affine(AA, BB, QQ, RR, SS, QT_LQR, T, x0_lqr, qq, rr, qqf)[2] # Use the same LQR affine solver with zero affine terms,
                                                                                        # to reconduct to the special case of standard Quadratic LQR

    # Simulate the system
    xx_sim = np.zeros((nx, T))
    uu_sim = np.zeros((nu, T))

    xx_sim[:, 0] = x0_lqr

    for tt in range(T - 1):
        # Control Law: u = u_ref + K * (x - x_ref)
        delta_x = xx_sim[:, tt] - xx_star[:, tt]
        delta_u = KK_reg[:, :, tt] @ delta_x
        
        uu_sim[:, tt] = uu_star[:, tt] + delta_u
        
        # Non-linear Dynamics integration
        xx_sim[:, tt+1] = dyn.rk4_step(xx_sim[:, tt], uu_sim[:, tt])
    
    # Fix last input for plotting (Zero Order Hold)
    uu_sim[:, -1] = uu_sim[:, -2] 

    # PLOTTING & ANIMATION
    time_axis = np.arange(T) * dt

    if cfg.TASK_3_PLOT_RESULTS:
        plotter.plot_results_task3(time_axis, xx_star, uu_star, xx_sim, uu_sim)

    # Animating the results
    if cfg.TASK_3_ANIMATION:
        ani = an.animate_task_trajectory(time_axis, xx_sim, xx_star, task_id=3)
        plt.tight_layout()
        plt.show()


# Definition of task 4 function
def task_4(xx_star, uu_star, x0):
    """
    TASK 4: TRAJECTORY TRACKING VIA MPC
    Uses Model Predictive Control to track the trajectory while respecting
    physical constraints (torque limits and angle limits).
    """

    print("\n==============================")
    print("Task 4 - TRACKING VIA MPC ")
    print("==============================\n")

    # SETUP MATRICES & LINEARIZATION
    # Linearize dynamics around the Task 2 optimal trajectory (LTV System)
    print("Linearizing dynamics around the optimal trajectory...")
    
    A = np.zeros((nx, nx, T))
    B = np.zeros((nx, nu, T))
    
    for t in range(T):
        dfx, dfu = dyn.get_linearization_np(xx_star[:, t], uu_star[:, t])
        A[:, :, t] = dfx
        B[:, :, t] = dfu

    # MPC CONFIGURATION
    Qt = cfg.Qt_task4
    Rt = cfg.Rt_task4
    Qf = cfg.QT_task4
    
    T_pred = int(cfg.T_PRED / dt) # Prediction horizon steps
    
    # Constraints dictionary
    constraints = {
        'u_min': np.array([cfg.U_MIN]),
        'u_max': np.array([cfg.U_MAX]),
        'x_min': np.array([cfg.THETA1_MIN, cfg.THETA2_MIN, cfg.THETA1_DOT_MIN, cfg.THETA2_DOT_MIN]),
        'x_max': np.array([cfg.THETA1_MAX, cfg.THETA2_MAX, cfg.THETA1_DOT_MAX, cfg.THETA2_DOT_MAX])
    }

    # SIMULATION INIT
    xx_sim = np.zeros((nx, T))
    uu_sim = np.zeros((nu, T))
    
    # Apply initial perturbation
    x0_perturbed = x0 + cfg.TASK_4_OFFSET
    xx_sim[:, 0] = x0_perturbed
    
    print(f"Starting MPC Tracking Loop (Horizon: {T_pred}, Perturbation: {cfg.TASK_4_OFFSET})...")
    
    # TRACKING LOOP
    for t in range(T - 1):
        # State measurement and deviation computation (Feedback)
        x_mpc = xx_sim[:, t]
        x_ref_t   = xx_star[:, t]
        delta_x0 = x_mpc - x_ref_t # Current tracking error (This is the initial condition for the MPC solver: Delta{x_0})

        current_horizon = min(T_pred, T - t - 1)

        if current_horizon < 1:
            uu_sim[:, t] = uu_star[:, t] # Last step (End of simulation)

        # Extract sequences for the predictive horizon
        A_seq = A[:, :, t : t + current_horizon]
        B_seq = B[:, :, t : t + current_horizon]
        u_ref_seq = uu_star[:, t : t + current_horizon]
        x_ref_seq = xx_star[:, t : t + current_horizon + 1]

        # If we are near the end, we might need to pad to maintain T_pred shape
        if current_horizon< T_pred:

            # Create full size arrays
            A_full = np.zeros((nx, nx, T_pred))
            B_full = np.zeros((nx, nu, T_pred))
            u_ref_full = np.zeros((nu, T_pred))
            x_ref_full = np.zeros((nx, T_pred + 1))

            # Fill the known parts
            A_full[:, :, :current_horizon] = A_seq
            B_full[:, :, :current_horizon] = B_seq
            u_ref_full[:, :current_horizon] = u_ref_seq
            x_ref_full[:, :current_horizon+1] = x_ref_seq

            # Fill the complemetary part with the last value
            # When the prediction horizon exceeds the remaining time steps,
            # we assume the system remains at the last known linearization and reference. 
            A_full[:, :, current_horizon:] = A[:, :, -1][:, :, np.newaxis]
            B_full[:, :, current_horizon:] = B[:, :, -1][:, :, np.newaxis]
            u_ref_full[:, current_horizon:] = uu_star[:, -1][:, np.newaxis]
            x_ref_full[:, current_horizon+1:] = xx_star[:, -1][:, np.newaxis]

            # Use the full padded versions
            A_seq, B_seq = A_full, B_full
            u_ref_seq, x_ref_seq = u_ref_full, x_ref_full
            
        # Call to the MPC Solver (CasADi)
        delta_u, feasible = mpc.ltv_mpc(
            A_seq, B_seq, Qt, Rt, Qf, 
            delta_x0, u_ref_seq, x_ref_seq, constraints
        )

        if not feasible:
            print(f"Warning: MPC infeasible at step {t}")
        
        # First Input application (u_real = u_ref + delta_u_opt)
        uu_sim[:, t] = uu_star[:, t] + delta_u 

        # Simulation Progress (Real Nonlinear Dynamics)
        xx_sim[:, t+1] = dyn.rk4_step(xx_sim[:, t], uu_sim[:, t])
        
        # Progress bar
        if t % 10 == 0:
            print(f"Simulating step {t}/{T}", end='\r')

    print(f"Simulating step {T}/{T} - Done.")
    
    # Fix last input for plotting (Zero Order Hold)
    uu_sim[:, -1] = uu_sim[:, -2]

    # PLOTTING & ANIMATION
    time_axis = np.arange(T) * dt
    
    if cfg.TASK_4_PLOT_RESULTS:
        print("Generating Task 4 plots...")
        plotter.plot_results_task4(time_axis, xx_star, uu_star, xx_sim, uu_sim)
        
    if cfg.TASK_4_ANIMATION:
        print("Starting Task 4 animation...")
        try:
            ani = an.animate_task_trajectory(time_axis, xx_sim, xx_star, task_id=4)
        except:
            pass
        plt.show()



# Main function
if __name__ == "__main__":
    # Initialize shared variables to None to trace if they have been computed
    xx_star, uu_star, x0_ref = None, None, None

    # Execute Task 0
    task_0()

    # Execute Task 1
    if cfg.TASK_1_ENABLE:
        task_1() 

    # Execute Task 2
    if cfg.TASK_2_ENABLE:
        # This task provides the reference trajectory for Task 3 and Task 4
        xx_star, uu_star, x0_ref = task_2()

    # Execute Task 3
    if cfg.TASK_3_ENABLE:
        # Check if we have the reference trajectory from Task 2
        if xx_star is not None and uu_star is not None:
            task_3(xx_star, uu_star, x0_ref)
        else:
            print("\n[ERROR] Cannot execute Task 3: Missing reference trajectory.\n")

    # Execute Task 4
    if cfg.TASK_4_ENABLE:
        if xx_star is not None:
            task_4(xx_star, uu_star, x0_ref)
            
        else:
            print("\n[ERROR] Cannot execute Task 4: Missing reference trajectory.\n")

    # Giorgione O&C


    
