import numpy as np
import matplotlib.pyplot as plt
from config import Config

cfg = Config()

def plot_results_task1(time, xx_ref, uu_ref, xx_opt, uu_opt, JJ, descent, kk):
    """
    Plotting results using matplotlib functions including intermediate iterations in dashed line.
    Layout:
      -1: Positions (Theta1, Theta2)
      -2: Velocity (dTheta1, dTheta2)
      -3: Input (Tau)

    Args:
        time: Time vector.
        xx_ref, uu_ref: Reference trajectories for state and input.
        xx_opt, uu_opt: History of computed trajectories (3D tensor: state x time x iteration).
        JJ: History vector of the cost function values.
        descent: vector of the descent direction.
        kk: Index of the final iteration reached.

    Shows:
      - Some Intermediate iterations
      - Final Iteration
    """
    uu_plot = np.copy(uu_opt[:, :, kk])
    uu_plot[:, -1] = uu_plot[:, -2] 
    
    # --- FIGURE 1: Ref Vs Opt (State and Inputs) And Tracking Errors (only angle positions) ---

    # 1. Position Theta 1
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[0, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[0, :, kk]), 'b-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Position $\theta_1$ [deg]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Position Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 2. Position Theta 2
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[1, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[1, :, kk]), 'r-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Position $\theta_2$ [deg]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Position Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 3. Angular Velocity Theta 1
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[2, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[2, :, kk]), 'b-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Velocity $\dot{\theta}_1$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Veloctiy Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 4. Angular Velocity Theta 2
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[3, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[3, :, kk]), 'r-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Velocity $\dot{\theta}_2$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Veloctiy Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 5. Input Torque
    plt.figure(figsize=(10, 6))
    plt.plot(time, uu_ref[0, :], 'k--', label="Ref")
    plt.plot(time, uu_opt[0, :, kk], 'g-', linewidth=2.5, label=f"Optimal Input (final Iter {kk})")
    plt.ylabel(r'Torque $\tau$ [Nm]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Input Torque', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 6 Error Position Theta 1 and Theta 2
    plt.figure(figsize=(10, 6))
    
    err_pos1 = np.degrees(xx_opt[0, :, kk] - xx_ref[0, :])
    err_pos2 = np.degrees(xx_opt[1, :, kk] - xx_ref[1, :])
    
    plt.subplot(2, 1, 1)
    plt.plot(time, err_pos1, 'b-', label=r'Error $\theta_1$')
    plt.ylabel(r'Err $\theta_1$ [deg]')
    plt.title('Angle Position Tracking Errors')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, err_pos2, 'r-', label=r'Error $\theta_2$')
    plt.ylabel(r'Err $\theta_1$ [deg]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Select Intermediate Iterations to plot 
    iters_to_plot = [0,1,2,4,10,22,kk-1]
    #We need to plot only the ones available 
    iters_to_plot = [i for i in iters_to_plot if i < kk]

    # Colors for intermediate steps (fading)
    colors_inter = plt.cm.Blues(np.linspace(0.3, 0.8, len(iters_to_plot)))

    # --- FIGURE 2: Trajectories (State and Inputs - 3x2) ---
    plt.figure(figsize=(14, 12))
    
    # 1. Position Theta 1 (UP SX - 3x2 GRID, cell 1)
    plt.subplot(3, 2, 1)
    plt.plot(time, np.degrees(xx_ref[0, :]), 'k--', label="Ref")
    
    # Plot Intermediate Iterations
    for idx, i_iter in enumerate(iters_to_plot):
        label_str = f"Iter {i_iter}"
        plt.plot(time, np.degrees(xx_opt[0, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, label=label_str, alpha=0.7)
    
    # Plot Final Optimal
    plt.plot(time, np.degrees(xx_opt[0, :, kk]), 'b-', linewidth=2, label=f"Optimal (Iter {kk})")
    
    plt.ylabel(r'Position $\theta_1$ [deg]')
    plt.title('Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    
    # 2. Position Theta 2 (Ut DX - 3x2 GRID, cell 2)
    plt.subplot(3, 2, 2)
    plt.plot(time, np.degrees(xx_ref[1, :]), 'k--', label="Ref")

    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[1, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
        
    plt.plot(time, np.degrees(xx_opt[1, :, kk]), 'r-', linewidth=2.5, label="Optimal")

    plt.ylabel(r'Position $\theta_2$ [deg]')
    plt.title('Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 3. Velocity Theta 1 (CENTER SX - 3x2 GRID, cell 3)
    plt.subplot(3, 2, 3)
    plt.plot(time, np.degrees(xx_ref[2, :]), 'k--', label="Ref")
    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[2, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
    plt.plot(time, np.degrees(xx_opt[2, :, kk]), 'b-', linewidth=2, label="Opt")
    plt.ylabel(r'Velocity $\dot{\theta}_1$ [deg/s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 4. Velocity dTheta 2 (CENTER DX - 3x2 GRID, cell 4)
    plt.subplot(3, 2, 4)
    plt.plot(time, np.degrees(xx_ref[3, :]), 'k--', label="Ref")
    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[3, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
    plt.plot(time, np.degrees(xx_opt[3, :, kk]), 'r-', linewidth=2, label="Opt")
    plt.ylabel(r'Velocity $\dot{\theta}_2$ [deg/s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 5. Input (DOWN - 3x1 GRID, cell 3)
    plt.subplot(3, 1, 3)
    plt.step(time, uu_ref[0, :], 'k--', where='post', label="Input Ref")
    
    # Plot Intermediate Inputs
    for idx, i_iter in enumerate(iters_to_plot):
        plt.step(time, uu_opt[0, :, i_iter], '--', where='post', color=colors_inter[idx], linewidth=1, alpha=0.7)
        
    plt.step(time, uu_opt[0, :, kk], 'g-', where='post', linewidth=2.5, label="Optimal Input")

    plt.ylabel(r'Torque $\tau$ [Nm]')
    plt.xlabel('Time [s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    # --- FIGURE 3: Convergence ---
    plt.figure(figsize=(12, 5))
    
    # Preparation of data for convergence
    iterations = np.arange(kk + 1)
    costs = JJ[:kk + 1]
    
    # Adding a small epsilon to avoid log(0) guaranteeing continuity 
    eps = 1e-16
    descent_valid = np.abs(descent[:kk + 1]) + eps
    
    # 1. Cost Function
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, costs, 'o-', color='purple', linewidth=2, markersize=6)
    plt.title(r"Reduced Cost Function (semilog scale)", fontweight='bold')
    plt.xlabel("Iteration k")
    plt.ylabel(r"$J(\mathbf{u}^k)$")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    
    # 2. Descent Direction
    plt.subplot(1, 2, 2)
    plt.semilogy(iterations, descent_valid, 's-', color='orange', linewidth=2, markersize=6)
    plt.title(r"Norm of Descent Direction (semilog scale)", fontweight='bold')
    plt.xlabel("Iteration k")
    plt.ylabel(r"$|\nabla J(\mathbf{u}^k)^T \Delta \mathbf{u^k}|$")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_results_task2(time, xx_ref, uu_ref, xx_opt, uu_opt, JJ, descent, kk):
    """
    Plotting results using matplotlib functions including intermediate iterations in dashed line.
    Layout:
      -1: Positions (Theta1, Theta2)
      -2: Velocity (dTheta1, dTheta2)
      -3: Input (Tau)

    Args:
        time: Time vector.
        xx_ref, uu_ref: Reference trajectories for state and input.
        xx_opt, uu_opt: History of computed trajectories (3D tensor: state x time x iteration).
        JJ: History vector of the cost function values.
        descent: vector of the descent direction.
        kk: Index of the final iteration reached.
        
    Shows:
      - Some Intermediate iterations
      - Final Iteration
    """
    
    uu_plot = np.copy(uu_opt[:, :, kk])
    uu_plot[:, -1] = uu_plot[:, -2] 
    
    # --- FIGURE 1: Ref Vs Opt (State and Inputs) And Tracking Errors (only angle positions) ---

    # 1. Position Theta 1
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[0, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[0, :, kk]), 'b-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Position $\theta_1$ [deg]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Position Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 2. Position Theta 2
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[1, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[1, :, kk]), 'r-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Position $\theta_2$ [deg]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Position Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 3. Angular Velocity Theta 1
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[2, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[2, :, kk]), 'b-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Velocity $\dot{\theta}_1$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Veloctiy Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 4. Angular Velocity Theta 2
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.degrees(xx_ref[3, :]), 'k--', label="Ref")
    plt.plot(time, np.degrees(xx_opt[3, :, kk]), 'r-', linewidth=2, label=f"Optimal (final Iter {kk})")
    plt.ylabel(r'Velocity $\dot{\theta}_2$ [deg/s]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Angular Veloctiy Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 5. Input Torque
    plt.figure(figsize=(10, 6))
    plt.plot(time, uu_ref[0, :], 'k--', label="Ref")
    plt.plot(time, uu_opt[0, :, kk], 'g-', linewidth=2.5, label=f"Optimal Input (final Iter {kk})")
    plt.ylabel(r'Torque $\tau$ [Nm]')
    plt.xlabel('Time [s]')
    plt.title('REFERENCE CURVE Vs OPTIMAL TRAJECTORY: Input Torque', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 6 Error Position Theta 1 and Theta 2
    plt.figure(figsize=(10, 6))
    
    err_pos1 = np.degrees(xx_opt[0, :, kk] - xx_ref[0, :])
    err_pos2 = np.degrees(xx_opt[1, :, kk] - xx_ref[1, :])
    
    plt.subplot(2, 1, 1)
    plt.plot(time, err_pos1, 'b-', label=r'Error $\theta_1$')
    plt.ylabel(r'Err $\theta_1$ [deg]')
    plt.title('Angle Position Tracking Errors')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time, err_pos2, 'r-', label=r'Error $\theta_2$')
    plt.ylabel(r'Err $\theta_1$ [deg]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    # Select Intermediate Iterations to plot 
    iters_to_plot = [0,1,2,kk-1]
    # We need to plot only the ones available 
    iters_to_plot = [i for i in iters_to_plot if i < kk]

    # Colors for intermediate steps (fading)
    colors_inter = plt.cm.Blues(np.linspace(0.3, 0.8, len(iters_to_plot)))

    # --- FIGURE 2: Trajectories (State and Inputs - 3x2) ---
    plt.figure(figsize=(14, 12))
    
    # 1. Position Theta 1 (UP SX - 3x2 GRID, cell 1)
    plt.subplot(3, 2, 1)
    plt.plot(time, np.degrees(xx_ref[0, :]), 'k--', label="Ref")
    
    # Plot Intermediate Iterations
    for idx, i_iter in enumerate(iters_to_plot):
        label_str = f"Iter {i_iter}"
        plt.plot(time, np.degrees(xx_opt[0, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, label=label_str, alpha=0.7)
    
    # Plot Final Optimal
    plt.plot(time, np.degrees(xx_opt[0, :, kk]), 'b-', linewidth=2, label=f"Optimal (Iter {kk})")
    plt.ylabel(r'Position $\theta_1$ [deg]')
    plt.title('Link 1', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    
    # 2. Position Theta 2 (Ut DX - 3x2 GRID, cell 2)
    plt.subplot(3, 2, 2)
    plt.plot(time, np.degrees(xx_ref[1, :]), 'k--', label="Ref")

    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[1, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
        
    plt.plot(time, np.degrees(xx_opt[1, :, kk]), 'r-', linewidth=2.5, label="Optimal")
    plt.ylabel(r'Position $\theta_2$ [deg]')
    plt.title('Link 2', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 3. Velocity dTheta 1 (CENTER SX - 3x2 GRID, cell 3)
    plt.subplot(3, 2, 3)
    plt.plot(time, np.degrees(xx_ref[2, :]), 'k--', label="Ref")
    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[2, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
    plt.plot(time, np.degrees(xx_opt[2, :, kk]), 'b-', linewidth=2, label="Opt")
    plt.ylabel(r'Velocity $\dot{\theta}_1$ [deg/s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 4. Velocity dTheta 2 (CENTER DX - 3x2 GRID, cell 4)
    plt.subplot(3, 2, 4)
    plt.plot(time, np.degrees(xx_ref[3, :]), 'k--', label="Ref")
    for idx, i_iter in enumerate(iters_to_plot):
        plt.plot(time, np.degrees(xx_opt[3, :, i_iter]), '--', color=colors_inter[idx], linewidth=1, alpha=0.7)
    plt.plot(time, np.degrees(xx_opt[3, :, kk]), 'r-', linewidth=2, label="Opt")
    plt.ylabel(r'Velocity $\dot{\theta}_2$ [deg/s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')

    # 5. Input (DOWN - 3x1 GRID, cell 3)
    plt.subplot(3, 1, 3)
    plt.step(time, uu_ref[0, :], 'k--', where='post', label="Input Ref")
    
    # Plot Intermediate Inputs
    for idx, i_iter in enumerate(iters_to_plot):
        plt.step(time, uu_opt[0, :, i_iter], '--', where='post', color=colors_inter[idx], linewidth=1, alpha=0.7)
        
    plt.step(time, uu_opt[0, :, kk], 'g-', where='post', linewidth=2.5, label="Optimal Input")

    plt.ylabel(r'Torque $\tau$ [Nm]')
    plt.xlabel('Time [s]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    # --- FIGURE 3: Convergence ---
    plt.figure(figsize=(12, 5))
    
    # Preparation of data for convergence
    iterations = np.arange(kk + 1)
    costs = JJ[:kk + 1]
    
    # Adding a small epsilon to avoid log(0) guaranteeing continuity 
    eps = 1e-16
    descent_valid = np.abs(descent[:kk + 1]) + eps
    
    # 1. Cost Function
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, costs, 'o-', color='purple', linewidth=2, markersize=6)
    plt.title(r"Reduced Cost Function (semilog scale)", fontweight='bold')
    plt.xlabel("Iteration k")
    plt.ylabel(r"$J(\mathbf{u}^k)$")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    
    # 2. Descent Direction
    plt.subplot(1, 2, 2)
    plt.semilogy(iterations, descent_valid, 's-', color='orange', linewidth=2, markersize=6)
    plt.title(r"Norm of Descent Direction (semilog scale)", fontweight='bold')
    plt.xlabel("Iteration k")
    plt.ylabel(r"$|\nabla J(\mathbf{u}^k)^T \Delta \mathbf{u^k}|$")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_results_task3(time, xx_ref, uu_ref, xx_sim, uu_sim):
    """
    Plot comparison Reference (Task 2 Opt) vs LQR Tracking (Task 3).
    Layout matches Task 4 but without constraint lines.
    Args:
        time: Time vector.
        xx_ref, uu_ref: Reference trajectories for state and input.
        xx_sim, uu_sim: Simulated trajectories (closed-loop state and input).
        
    """
    
    # --- FIGURE 1: STATE TRACKING (Pos & Vel) & ERRORS ---
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('Task 3: LQR Tracking Performance (States)', fontsize=16, fontweight='bold')

    # --- 1.1 Theta 1 Position ---
    ax = axs[0, 0]
    ax.plot(time, np.degrees(xx_ref[0, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[0, :]), 'b-', linewidth=2, label='LQR Link 1')
    ax.set_ylabel(r'Pos $\theta_1$ [deg]')
    ax.set_title('Link 1 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 1.2 Theta 2 Position ---
    ax = axs[0, 1]
    ax.plot(time, np.degrees(xx_ref[1, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[1, :]), 'r-', linewidth=2, label='LQR Link 2')
    ax.set_ylabel(r'Pos $\theta_2$ [deg]')
    ax.set_title('Link 2 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 2.1 Error Theta 1 ---
    ax = axs[1, 0]
    err_pos1 = np.degrees(xx_sim[0,:] - xx_ref[0,:])
    ax.plot(time, err_pos1, color='dodgerblue', linewidth=1.5, label=r'Error $\theta_1$')
    ax.set_ylabel(r'Err $\theta_1$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 2.2 Error Theta 2 ---
    ax = axs[1, 1]
    err_pos2 = np.degrees(xx_sim[1,:] - xx_ref[1,:])
    ax.plot(time, err_pos2, color='tomato', linewidth=1.5, label=r'Error $\theta_2$')
    ax.set_ylabel(r'Err $\theta_2$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 3.1 Velocity Theta 1 ---
    ax = axs[2, 0]
    ax.plot(time, np.degrees(xx_ref[2, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[2, :]), 'b-', linewidth=2, label='LQR Vel 1')
    ax.set_ylabel(r'Vel $\dot{\theta}_1$ [deg/s]')
    ax.set_title('Link 1 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 3.2 Velocity Theta 2 ---
    ax = axs[2, 1]
    ax.plot(time, np.degrees(xx_ref[3, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[3, :]), 'r-', linewidth=2, label='LQR Vel 2')
    ax.set_ylabel(r'Vel $\dot{\theta}_2$ [deg/s]')
    ax.set_title('Link 2 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 4.1 Error Vel 1 ---
    ax = axs[3, 0]
    err_vel1 = np.degrees(xx_sim[2,:] - xx_ref[2,:])
    ax.plot(time, err_vel1, color='dodgerblue', linewidth=1.5, label=r'Error $\dot{\theta}_1$')
    ax.set_ylabel(r'Err $\dot{\theta}_1$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 4.2 Error Vel 2 ---
    ax = axs[3, 1]
    err_vel2 = np.degrees(xx_sim[3,:] - xx_ref[3,:])
    ax.plot(time, err_vel2, color='tomato', linewidth=1.5, label=r'Error $\dot{\theta}_2$')
    ax.set_ylabel(r'Err $\dot{\theta}_2$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # --- FIGURE 2: INPUT & INPUT ERROR ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    uu_ref_plot = np.copy(uu_ref); uu_ref_plot[:,-1] = uu_ref_plot[:,-2]
    uu_sim_plot = np.copy(uu_sim); uu_sim_plot[:,-1] = uu_sim_plot[:,-2]

    # 1. Input Comparison
    ax = axs2[0]
    ax.step(time, uu_ref_plot[0, :], 'k--', where='post', label='Ref Input', alpha=0.6)
    ax.step(time, uu_sim_plot[0, :], 'g-', where='post', linewidth=2, label='LQR Input')
    ax.set_ylabel(r'Torque [Nm]')
    ax.set_title('Control Input', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 2. Input Error
    ax = axs2[1]
    err_u = uu_sim_plot[0, :] - uu_ref_plot[0, :]
    ax.step(time, err_u, 'seagreen', where='post', label=r'$\Delta u$')
    ax.set_ylabel(r'Err $\tau$ [Nm]')
    ax.set_xlabel('Time [s]')
    ax.set_title(r'Input Deviation ($\Delta u$)', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_results_task4(time, xx_ref, uu_ref, xx_sim, uu_sim):
    """
    Plots comparison Reference vs MPC Tracking with Enhanced Layout & Colors.
    Detailed legends included for constraints.
    Args:
        time: Time vector.
        xx_ref, uu_ref: Reference trajectories for state and input.
        xx_sim, uu_sim: Simulated trajectories (closed-loop state and input).
    """
    
    # --- FIGURE 1: STATES AND VELOCITY ---
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('Task 4: MPC Tracking Performance (States)', fontsize=16, fontweight='bold')

    # Convert Constraints to Degrees
    th1_max = np.degrees(cfg.THETA1_MAX); th1_min = np.degrees(cfg.THETA1_MIN)
    th2_max = np.degrees(cfg.THETA2_MAX); th2_min = np.degrees(cfg.THETA2_MIN)
    dth1_max = np.degrees(cfg.THETA1_DOT_MAX); dth1_min = np.degrees(cfg.THETA1_DOT_MIN)
    dth2_max = np.degrees(cfg.THETA2_DOT_MAX); dth2_min = np.degrees(cfg.THETA2_DOT_MIN)

    # --- 1.1 Theta 1 ---
    ax = axs[0, 0]
    if cfg.TASK_4_CONSTRAINTS_ON:
        ax.fill_between(time, th1_min, th1_max, color='tab:green', alpha=0.1, label='Feasible Zone')
        ax.axhline(th1_max, color='darkred', linestyle='--', linewidth=1.5, label=r'$\theta_{1,max}$')
        ax.axhline(th1_min, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$\theta_{1,min}$')
    
    ax.plot(time, np.degrees(xx_ref[0, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[0, :]), 'b-', linewidth=2, label='MPC Link 1')
    
    ax.set_ylabel(r'Pos $\theta_1$ [deg]')
    ax.set_title('Link 1 Position', fontweight='bold')
    ax.set_ylim([th1_min - 10, th1_max + 10])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2)

    # --- 1.2 Theta 2 ---
    ax = axs[0, 1]
    if cfg.TASK_4_CONSTRAINTS_ON:
        ax.fill_between(time, th2_min, th2_max, color='tab:green', alpha=0.1, label='Feasible Zone')
        ax.axhline(th2_max, color='darkred', linestyle='--', linewidth=1.5, label=r'$\theta_{2,max}$')
        ax.axhline(th2_min, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$\theta_{2,min}$')

    ax.plot(time, np.degrees(xx_ref[1, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[1, :]), 'r-', linewidth=2, label='MPC Link 2')
    
    ax.set_ylabel(r'Pos $\theta_2$ [deg]')
    ax.set_title('Link 2 Position', fontweight='bold')
    ax.set_ylim([th2_min - 10, th2_max + 10])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2)

    # --- 2.1 Error Theta 1 ---
    ax = axs[1, 0]
    err_pos1 = np.degrees(xx_sim[0,:] - xx_ref[0,:])
    ax.plot(time, err_pos1, color='dodgerblue', linewidth=1.5, label=r'Error $\theta_1$')
    ax.set_ylabel(r'Err $\theta_1$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 2.2 Error Theta 2 ---
    ax = axs[1, 1]
    err_pos2 = np.degrees(xx_sim[1,:] - xx_ref[1,:])
    ax.plot(time, err_pos2, color='tomato', linewidth=1.5, label=r'Error $\theta_2$')
    ax.set_ylabel(r'Err $\theta_2$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 3.1 Velocity Theta 1 ---
    ax = axs[2, 0]
    if cfg.TASK_4_CONSTRAINTS_ON:
        ax.fill_between(time, dth1_min, dth1_max, color='tab:green', alpha=0.1, label='Feasible')
        ax.axhline(dth1_max, color='darkred', linestyle='--', linewidth=1.5, label=r'$\dot{\theta}_{1,max}$')
        ax.axhline(dth1_min, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$\dot{\theta}_{1,min}$')

    ax.plot(time, np.degrees(xx_ref[2, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[2, :]), 'b-', linewidth=2, label='MPC Vel 1')
    
    ax.set_ylabel(r'Vel $\dot{\theta}_1$ [deg/s]')
    ax.set_title('Link 1 Velocity', fontweight='bold')
    ax.set_ylim([dth1_min - 20, dth1_max + 20])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2)

    # --- 3.2 Velocity Theta 2 ---
    ax = axs[2, 1]
    if cfg.TASK_4_CONSTRAINTS_ON:
        ax.fill_between(time, dth2_min, dth2_max, color='tab:green', alpha=0.1, label='Feasible')
        ax.axhline(dth2_max, color='darkred', linestyle='--', linewidth=1.5, label=r'$\dot{\theta}_{2,max}$')
        ax.axhline(dth2_min, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$\dot{\theta}_{2,min}$')

    ax.plot(time, np.degrees(xx_ref[3, :]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[3, :]), 'r-', linewidth=2, label='MPC Vel 2')
    
    ax.set_ylabel(r'Vel $\dot{\theta}_2$ [deg/s]')
    ax.set_title('Link 2 Velocity', fontweight='bold')
    ax.set_ylim([dth2_min - 20, dth2_max + 20])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize='small', ncol=2)

    # --- 4.1 Error Vel 1 ---
    ax = axs[3, 0]
    err_vel1 = np.degrees(xx_sim[2,:] - xx_ref[2,:])
    ax.plot(time, err_vel1, color='dodgerblue', linewidth=1.5, label=r'Error $\dot{\theta}_1$')
    ax.set_ylabel(r'Err $\dot{\theta}_1$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 4.2 Error Vel 2 ---
    ax = axs[3, 1]
    err_vel2 = np.degrees(xx_sim[3,:] - xx_ref[3,:])
    ax.plot(time, err_vel2, color='tomato', linewidth=1.5, label=r'Error $\dot{\theta}_2$')
    ax.set_ylabel(r'Err $\dot{\theta}_2$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # --- FIGURE 2: INPUT AND INPUT ERROR ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    
    uu_ref_plot = np.copy(uu_ref); uu_ref_plot[:,-1] = uu_ref_plot[:,-2]
    uu_sim_plot = np.copy(uu_sim); uu_sim_plot[:,-1] = uu_sim_plot[:,-2]

    # 1. Input Comparison
    ax = axs2[0]
    # Fill Feasible Zone
    if cfg.TASK_4_CONSTRAINTS_ON:
        ax.fill_between(time, cfg.U_MIN, cfg.U_MAX, color='tab:green', alpha=0.1, label='Feasible')
        ax.axhline(cfg.U_MAX, color='darkred', linestyle='--', linewidth=1.5, label=r'$u_{max}$')
        ax.axhline(cfg.U_MIN, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$u_{min}$')

    ax.step(time, uu_ref_plot[0, :], 'k--', where='post', label='Ref', alpha=0.6)
    ax.step(time, uu_sim_plot[0, :], 'g-', where='post', linewidth=2, label='MPC Input')
    
    ax.set_ylabel(r'Torque [Nm]')
    ax.set_title('Control Input (Constrained)', fontweight='bold')
    ax.set_ylim([cfg.U_MIN - 5, cfg.U_MAX + 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=3)

    # 2. Input Error
    ax = axs2[1]
    err_u = uu_sim_plot[0, :] - uu_ref_plot[0, :]
    ax.step(time, err_u, 'seagreen', where='post', label=r'$\Delta u$')
    ax.set_ylabel(r'Err $\tau$ [Nm]')
    ax.set_xlabel('Time [s]')
    ax.set_title(r'Input Deviation ($\Delta u$)', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
