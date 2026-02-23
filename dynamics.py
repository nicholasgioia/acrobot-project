import casadi as ca
import numpy as np
from config import Config

class AcrobotDynamics:
    def __init__(self):
        self.cfg = Config()
        
        # Initialize CasADi symbolic dynamics
        self._setup_symbolic_dynamics()

    def _setup_symbolic_dynamics(self):
        """
        Defines the Acrobot's mathematical model using symbolic variables.
        Calculates equations of motion, discretization, and Jacobians.
        """
        # Define symbolic variables
        # State x = [th1, th2, dth1, dth2]
        x = ca.SX.sym('x', self.cfg.nx)
        u = ca.SX.sym('u', self.cfg.nu) # u = [tau]

        q1 = x[0]
        q2 = x[1]
        dq1 = x[2]
        dq2 = x[3]

        m1, m2 = self.cfg.m1, self.cfg.m2
        l1, l2 = self.cfg.l1, self.cfg.l2
        lc1, lc2 = self.cfg.lc1, self.cfg.lc2
        I1, I2 = self.cfg.I1, self.cfg.I2
        f1, f2 = self.cfg.f1, self.cfg.f2
        g = self.cfg.g

        # Inertia matrix M(q)
        # M11 = I1 + I2 + m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2))
        M11 = I1 + I2 + m1*(lc1**2) + m2*(l1**2 + lc2**2 + 2*l1*lc2*ca.cos(q2))
        # M12 = I2 + m2*lc2*(l1*cos(q2) + lc2)
        M12 = I2 + m2*lc2*(l1*ca.cos(q2) + lc2)
        M21 = M12
        # M22 = I2 + m2*lc2^2
        M22 = I2 + m2*(lc2**2)

        M = ca.vertcat(
            ca.horzcat(M11, M12),
            ca.horzcat(M21, M22)
        )

        # Coriolis matrix C(q, dq)
        h = m2*l1*lc2*ca.sin(q2)
        C11 = -h * dq2
        C12 = -h * (dq1 + dq2)
        C21 = h * dq1
        C22 = 0.0

        C = ca.vertcat(
            ca.horzcat(C11, C12),
            ca.horzcat(C21, C22)
        )

        # Gravity vector G(q)
        # G1 = g*m1*lc1*sin(q1) + g*m2*l1*sin(q1) + lc2*sin(q1+q2)
        G1 = g*m1*lc1*ca.sin(q1) + g*m2*l1*ca.sin(q1) + lc2*ca.sin(q1+q2)
        # G2 = g*m2*lc2*sin(q1+q2)
        G2 = g*m2*lc2*ca.sin(q1+q2)

        G = ca.vertcat(G1, G2)

        # Friction matrix F
        # F is diagonal: F*dq = [f1*dq1; f2*dq2]
        F_term = ca.vertcat(f1*dq1, f2*dq2)

        # Control input: M * ddq + C * dq + F * dq + G = Tau_vec
        # Tau vector: [0; u], motor only on second joint
        tau_vec = ca.vertcat(0, u[0])

        # Solve for angular acceleration: ddq = inv(M) * (tau - C*dq - F*dq - G)
        dq = ca.vertcat(dq1, dq2)
        rhs = tau_vec - ca.mtimes(C, dq) - F_term - G # right-hand side

        ddq = ca.solve(M, rhs)

        # f_continuous = dx/dt = [dq; ddq]
        x_dot = ca.vertcat(dq, ddq)
        
        # CasaDi function for continuous dynamics: x_dot = f(x, u)
        self.f_cont = ca.Function('f_cont', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

        # Discretization RK4 (Runge-Kutta 4) [Task 0]
        dt = self.cfg.dt
        k1 = self.f_cont(x, u)
        k2 = self.f_cont(x + 0.5*dt*k1, u)
        k3 = self.f_cont(x + 0.5*dt*k2, u)
        k4 = self.f_cont(x + dt*k3, u)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # CasaDi funxtion for discrete dynamics: x_{k+1} = F(x_k, u_k)
        self.F_discrete = ca.Function('F_discrete', [x, u], [x_next], ['x', 'u'], ['x_next'])

        # Linearization (Jacobians A, B)
        # A = df/dx, B = df/du
        jac_A = ca.jacobian(x_next, x)
        jac_B = ca.jacobian(x_next, u)
        
        self.get_jacobians = ca.Function('J_discrete', [x, u], [jac_A, jac_B], ['x', 'u'], ['A', 'B'])

    def rk4_step(self, x0, u0):
        """Evaluates one discrete time step forward using RK4."""

        res = self.F_discrete(x0, u0)
        return np.array(res).flatten()

    def get_linearization_np(self, x0, u0):
        """Returns the A and B matrices as NumPy arrays for a given state and input."""

        A_val, B_val = self.get_jacobians(x0, u0)
        return np.array(A_val), np.array(B_val)
    
    def get_gravity_terms(self, theta_1, theta_2):
        """Returns the gravity forces (G vector) specifically for equilibrium analysis."""
        
        m1, m2 = self.cfg.m1, self.cfg.m2
        l1, lc2 = self.cfg.l1, self.cfg.lc2
        lc1 = self.cfg.lc1
        g = self.cfg.g

        s1 = np.sin(theta_1)
        s12 = np.sin(theta_1 + theta_2)

        G1 = (m1*lc1 + m2*l1)*g*s1 + m2*lc2*g*s12
        G2 = m2*lc2*g*s12

        return np.array([G1, G2])