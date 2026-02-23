import numpy as np
import matplotlib.pyplot as plt
from config import Config
import equilibrium as eq

class TrajectoryGenerator:
    def __init__(self, dynamics_model):
        self.dyn = dynamics_model
        self.cfg = Config()

        self.dt = self.cfg.dt
        self.nx = self.cfg.nx
        self.nu = self.cfg.nu
        self.T = self.cfg.T_horizon
        self.T_mid = self.T / 2
        self.N = self.cfg.N_steps


    def generate_step_trajectory(self, x_start, x_end, u_start, u_end):
        '''Generates a reference step from an equilibrium to another one given from equilibrium.py'''

        # Create and initialize the reference
        xx_ref = np.zeros((self.nx, self.N))
        uu_ref = np.zeros((self.nu, self.N))

        # Generate the theta_ref values during the transition
        for t in range(self.N):
            if t < self.N / 2:
                xx_ref[:, t] = x_start
                uu_ref[:, t] = u_start
            else:
                xx_ref[:, t] = x_end
                uu_ref[:, t] = u_end
        return xx_ref, uu_ref
    
    def poly5(self, t, T, q0, qf):
        """
        5th-order polynomial (Minimum Jerk).
        Returns consistent position and velocity at time t.
        """
        if t < 0:
            t = 0
        if t > T:
            t = T
        
        tau = t / T #normalized time in [0,1]

        # calculation of powers
        tau2 = tau**2
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5

        h = qf - q0

        # position
        q_t = q0 + h * (10 * tau3 - 15 * tau4 + 6 * tau5)

        # velocity
        dq_t = h * (30 * tau2 - 60 * tau3 + 30 * tau4) *  (1.0 / T)

        return q_t, dq_t
    
    def generate_polynomial_trajectory(self, x_start, x_end, u_start, u_end):
        '''Generates a reference trajectory using 5th order polynomials between two states.'''

        # Create and initialize the reference
        xx_ref = np.zeros((self.nx, self.N))
        uu_ref = np.zeros((self.nu, self.N))

        # Variable used to memorize the theta2 of previous step
        # Initialization with start value 
        prev_q1 = x_start[0]
        delta_t_transition = self.cfg.TASK_2_DELTA_T_TRANSITION
        t_start_transition = (self.T / 2) - (delta_t_transition / 2)
        t_end_transition = (self.T / 2) + (delta_t_transition / 2)

        for t in range(self.N):
            time = t * self.dt

            if time <= t_start_transition:
                # Initial fase: Initial equilibrium [0, 8]
                q2_t = x_start[1]
                dq2_t = 0.0
            
            elif time >= t_end_transition:
                # Terminal fase: End equilibrium [12, 20]
                q2_t = x_end[1]
                dq2_t = 0.0
            
            else:
                # Transition fase: Polynomial curve ]8, 12[
                t_local = time - t_start_transition
                q2_t, dq2_t = self.poly5(t_local, t_end_transition-t_start_transition, x_start[1], x_end[1])

            # Computations of velocity for Theta 2 (Finite Differences)
            # since pos_th2 it's not an analytical Polynomial we derive it numerically 

            q_eq, u_eq = eq.equilibrium_finding(q2_t)
            q1_t = q_eq[0] # Extract theta1 from equilibrium state

            if t == 0:
                dq1_t = 0.0
            else:
                dq1_t = (q1_t - prev_q1) / self.dt

            prev_q1 = q1_t

            xx_ref[0, t] = q1_t
            xx_ref[1, t] = q2_t
            xx_ref[2, t] = dq1_t
            xx_ref[3, t] = dq2_t

            # Input Interpolations
            uu_ref[0, t] = u_eq[0]  # Torque at joint 2 from equilibrium finding

        # Exact final position/input torque at the last step 
        xx_ref[:, -1] = x_end
        uu_ref[:, -1] = u_end
        # Velocity forced to zero at the last step
        xx_ref[2, -1] = 0.0
        xx_ref[3, -1] = 0.0

        return xx_ref, uu_ref