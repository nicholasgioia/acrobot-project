import numpy as np
from config import Config
from scipy.optimize import fsolve
from dynamics import AcrobotDynamics

cfg = Config() 
dyn =AcrobotDynamics()


# Extract parameters from Config
m1, m2 =cfg.m1, cfg.m2
l1, l2 = cfg.l1, cfg.l2
lc1, lc2 = cfg.lc1, cfg.lc2
I1, I2 = cfg.I1, cfg.I2
f1, f2 = cfg.f1, cfg.f2
g = cfg.g

# Simulation parameters
dt = cfg.dt
nx = cfg.nx
nu = cfg.nu


def equilibrium_finding(theta2_target):
    """
    Finds the equilibrium state and the required torque to maintain a specific
    second-joint angle (theta2) in a static position.
    """
    
    def dynamics_residual(z):
        """
        Helper function for the numerical solver.
        Calculates the difference (residual) between current accelerations and zero.
        """
        theta1_guess = z[0]
        tau_guess = z[1]
        
        # Build the trial state (velocities are zero for equilibrium)
        # x = [theta1, theta2, dtheta1, dtheta2]
        x_trial = np.array([theta1_guess, theta2_target, 0.0, 0.0])
        
        # Build the trial input
        u_trial = np.array([tau_guess])
        
        # Compute the C-T dynamics
        # f_cont returns x_dot = [dtheta1, dtheta2, ddtheta1, ddtheta2]
        x_dot_dm = dyn.f_cont(x_trial, u_trial)
        x_dot = np.array(x_dot_dm).flatten()
        
        # To be at equilibrium, also accelerations must be zero
        ddtheta1 = x_dot[2]
        ddtheta2 = x_dot[3]
        
        return [ddtheta1, ddtheta2]

    # Initial guess
    z_guess = [0.0, 0.0]

    # Fsolve (Newton-Raphson) to find the root of dynamics_residual
    z_solution = fsolve(dynamics_residual, z_guess)
    
    theta1_eq = z_solution[0]
    tau_eq = z_solution[1]

    print(f"Equilibrium found (Numeric): theta1={np.degrees(theta1_eq):.4f}, tau={tau_eq:.4f} for theta2={np.degrees(theta2_target):.4f}")

    # Pack the equilibrium state and input
    x_eq = np.array([theta1_eq, theta2_target, 0.0, 0.0])
    u_eq = np.array([tau_eq])
    
    return x_eq, u_eq


    