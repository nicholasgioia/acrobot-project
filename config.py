import numpy as np

class Config:
    # --- Physical parameters - Set 1 ---
    m1 = 1.0   # Mass of the first link [kg]
    m2 = 1.0   # Mass of the second link [kg]
    l1 = 1.0   # Total length of the first link [m]
    l2 = 1.0   # Total length of the second link [m]
    lc1 = 0.5  # Distance from the first joint to the center of mass of link 1 [m]
    lc2 = 0.5  # Distance from the second joint to the center of mass of link 2 [m]
    I1 = 0.33  # Moment of inertia of the first link [kg*m^2]
    I2 = 0.33  # Moment of inertia of the second link [kg*m^2]
    f1 = 1.0   # Viscous friction coefficient at joint 1
    f2 = 1.0   # Viscous friction coefficient at joint 2
    g = 9.81   # Acceleration due to gravity [m/s^2]
    
    # --- Simulation Parameters ---
    dt = 0.01                           # Discretization time step [s]
    T_horizon = 20.0                    # Time horizon for the simulation [s]
    N_steps = int(T_horizon / dt)       # Number of total simulation steps
    
    # --- Dimensions ---
    nx = 4 # [theta1, theta2, dtheta1, dtheta2]
    nu = 1 # [tau]

    # --- Armijo Parameters ---
    ARMIJO_PLOT_RESOLUTION = 20 # Number of steps for Armijo plots
    ARMIJO_C      = 0.5         # Tangent slope factor for sufficient decrease condition
    ARMIJO_BETA   = 0.7         # Reduction coedfficient for stepsize
    ARMIJO_MAX_IT = 20          # Max backtrack steps for Armijo iterations
    ARMIJO_STEPSIZE_0 = 1       # Initial value for Armijo Step Size when k=0
    ARMIJO_TERM_COND = 1e-6     # Condition to stop the search

    # -------------------------------------
    # TASK 0 
    # -------------------------------------

    # Value to check equilibrium point
    TASK_0_THETA2_START_DEG = np.radians(0.0)
    TASK_0_THETA2_END_DEG   = np.radians(180.0)


    # -------------------------------------
    # TASK 1 TRAJECTORY GENERATION
    # -------------------------------------

    # Set Task 1 enable
    TASK_1_ENABLE = False

    # Step reference values (Angles in radians for Theta2)
    TASK_1_THETA2_START_DEG = np.radians(0.0)
    TASK_1_THETA2_END_DEG   = np.radians(40.0)

    # Newton method parameters
    TASK_1_PLOT_ARMIJO = True                   
    TASK_1_ARMIJO_PLOT_NUMBER = 20
    TASK_1_MAX_ITERS = 100

    # Plots and Animation
    TASK_1_PLOT_RESULTS = True
    TASK_1_ANIMATION = True

    # Cost Matrices for Newton (Q, R, Qf)

    Qt_task1 = np.diag([5, 10, 0.1, 0.1])      # Stage Cost (Tracking Error) WE LEAVE FREEDOM TO OSCILLATE AND SWING-UP EVENTUALLY
    Rt_task1 = np.eye(nu) * 1                    # Actuator Cost of U
    QT_task1 = np.diag([5000, 10000, 500, 500])  # Terminal Cost (Final Accuracy) 

    # -------------------------------------
    # TASK 2 TRAJECTORY GENERATION
    # -------------------------------------

    # Set Task 2 enable
    TASK_2_ENABLE = True

    # Poly5 reference trajectory parameters
    TASK_2_THETA2_START_DEG = np.radians(0.0)
    TASK_2_THETA2_END_DEG   = np.radians(180.0)

    TASK_2_DELTA_T_TRANSITION = 4.0  # Duration of the transition phase [s]

    # Newton method parameters
    TASK_2_PLOT_ARMIJO = True                   
    TASK_2_ARMIJO_PLOT_NUMBER = 20
    TASK_2_MAX_ITERS = 100

    # Plots and Animation
    TASK_2_PLOT_RESULTS = True
    TASK_2_ANIMATION = True

    # Cost Matrices for Newton (Q, R, Qf)

    Qt_task2 = np.diag([1000, 1000, 0.1, 0.1])      # Stage Cost (Tracking Error) WE LEAVE FREEDOM TO OSCILLATE AND SWING-UP EVENTUALLY
    Rt_task2 = np.eye(nu) * 0.1                     # Actuator Cost of U
    QT_task2 = np.diag([10000, 10000, 500, 500])    # Terminal Cost (Final Accuracy) 

    # -------------------------------------
    # TASK 3 TRAJECTORY TRACKING VIA LQR 
    # -------------------------------------

    # Set Task 3 enable
    TASK_3_ENABLE = True

    # OFFSET to simulate the pertubation on initial state
    TASK_3_PERTURBATION=np.array([0.2, -0.2, 0.1, -0.1])

    # Plots and Animation
    TASK_3_PLOT_RESULTS = True
    TASK_3_ANIMATION = True

    # Cost Matrices for Newton (Q, R, Qf)
    Qt_task3 = np.diag([1000.0, 1000.0, 100.0, 100.0])      # Stage Cost (Tracking Error)
    Rt_task3 = np.eye(nu) * 0.001                           # Actuator Cost of U
    QT_task3 = np.diag([1000.0, 1000.0, 100.0, 100.0])      # Terminal Cost (Final Accuracy)


    # -------------------------------------
    # TASK 4 TRAJECTORY TRACKING VIA MPC
    # -------------------------------------

    # Set Task 4 enable
    TASK_4_ENABLE = True
    TASK_4_CONSTRAINTS_ON = True

    # OFFSET to simulate the pertubation on initial state
    TASK_4_OFFSET=np.array([0.2, -0.2, 0.1, -0.1]) 

    # MPC Parameters
    T_PRED = 0.1                          # MPC Prediction horizon
    U_MAX = 5                          # Torque/Input Saturation constraint [Nm]
    U_MIN = -20                
    THETA1_MAX = np.radians(20)         # Angles constraints [rad]
    THETA1_MIN = - THETA1_MAX
    THETA2_MAX = np.radians(200)
    THETA2_MIN = -THETA2_MAX
    THETA1_DOT_MAX = np.radians(60)     # Angular velocity constraints [rad/s]
    THETA1_DOT_MIN = - THETA1_DOT_MAX
    THETA2_DOT_MAX = np.radians(100)
    THETA2_DOT_MIN = -THETA2_DOT_MAX

    # Plots and Animation
    TASK_4_PLOT_RESULTS = True
    TASK_4_ANIMATION = True 

    # Cost Matrices for Newton (Q, R, Qf)
    Qt_task4 = np.diag([1000.0, 1000.0, 100.0, 100.0])      # Stage Cost (Tracking Error)
    Rt_task4 = np.eye(nu) * 0.001                           # Actuator Cost of U
    QT_task4 = np.diag([1000.0, 1000.0, 100.0, 100.0])      # Terminal Cost (Final Accuracy)

