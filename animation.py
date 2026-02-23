import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from config import Config

cfg = Config()

dt = cfg.dt
l1 = cfg.l1
l2 = cfg.l2

def animate_robot_dynamics(time, xx):
    """
    Creates a comprehensive animation with the Robot on the left
    and State trajectories (Theta1, Theta2) on the right.
    
    Parameters:
        time:   Time array [T,]
        xx:     State history [4, T]

    Returns:
        ani:    the animation
    """
    
    # 1. Setup Figure and Grid
    # Reduced height since we only have 2 plots on the right
    fig = plt.figure(figsize=(14, 6)) 
    # Use a fixed title or no title since the argument is removed
    fig.suptitle("Task 0", fontsize=16, fontweight='bold')
    
    # 2 rows instead of 4
    gs = GridSpec(2, 2, figure=fig)
    
    # --- Robot View (Left Column, spanning both rows) ---
    ax_robot = fig.add_subplot(gs[:, 0])
    ax_robot.set_aspect('equal')
    
    # Dynamic limits
    L_tot = l1 + l2 + 0.2
    ax_robot.set_xlim(-L_tot, L_tot)
    ax_robot.set_ylim(-L_tot, L_tot)
    ax_robot.grid(True, linestyle=':', alpha=0.6)
    ax_robot.set_title(r"Gymnast Robot Animation for Dynamic Validation", fontweight = 'bold')
    ax_robot.set_xlabel("X [m]")
    ax_robot.set_ylabel("Y [m]")
    
    # Robot Elements
    line_arm, = ax_robot.plot([], [], 'o-', lw=4, color='tab:blue', markersize=8, label='Links') 
    trace, = ax_robot.plot([], [], '-', color='red', alpha=0.5, lw=1) 
    
    time_text = ax_robot.text(0.05, 0.95, '', transform=ax_robot.transAxes, fontsize=12, 
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- Plots View (Right Column) ---
    
    # Plot 1: Theta 1 (Top Right)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(time, np.degrees(xx[0, :]), 'b-', lw=2)
    cursor1, = ax1.plot([], [], 'ro', zorder=10)
    ax1.set_ylabel(r'$\theta_1$ [deg]')
    ax1.grid(True, alpha=0.4)
    ax1.set_title(r"Angles Evolution", fontweight='bold', fontsize=10)
    
    # Plot 2: Theta 2 (Bottom Right)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.plot(time, np.degrees(xx[1, :]), 'r-', lw=2)
    cursor2, = ax2.plot([], [], 'bo', zorder=5)
    ax2.set_ylabel(r'$\theta_2$ [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True, alpha=0.4)
    
    # History for trace
    history_x, history_y = [], []
    
    # Animation Function
    skip = max(1, int(0.04 / dt)) 
    
    def init():
        line_arm.set_data([], [])
        trace.set_data([], [])
        return line_arm, trace, cursor1, cursor2

    def update(frame):
        idx = frame * skip
        if idx >= len(time): idx = len(time) - 1
        
        # --- Robot Update ---
        th1 = xx[0, idx]
        th2 = xx[1, idx]
        
        # Kinematics
        x0, y0 = 0, 0
        x1 = l1 * np.sin(th1)
        y1 = -l1 * np.cos(th1)
        x2 = x1 + l2 * np.sin(th1 + th2)
        y2 = y1 - l2 * np.cos(th1 + th2)
        
        line_arm.set_data([x0, x1, x2], [y0, y1, y2])
        
        # Trace
        history_x.append(x2)
        history_y.append(y2)
        if len(history_x) > 200: 
            history_x.pop(0); history_y.pop(0)
        trace.set_data(history_x, history_y)
        
        time_text.set_text(f'Time: {time[idx]:.2f} s')
        
        # --- Cursors Update ---
        t_curr = time[idx]
        cursor1.set_data([t_curr], [np.degrees(xx[0, idx])])
        cursor2.set_data([t_curr], [np.degrees(xx[1, idx])])
        
        return line_arm, trace, cursor1, cursor2, time_text

    # Create Animation
    n_frames = len(time) // skip
    ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                                  interval=40, blit=True)
    
    plt.tight_layout()
    return ani

def animate_task_trajectory(time, xx, xx_ref=None, task_id=None):
    
    """
    Create an animation of the gymnast robot and optionally animates cursors on trajectory graphs.
    
    Parameters:
        time:   array for times  [N,].
        xx:     actual trajectory
        xx_ref: reference that should be followed
        task_id=None: optional task identifier for title purposes

    Returns:
        ani:    the animation
    """
    
    # Link lenght parameters for the size plot
    L_tot = l1 + l2 + 0.5
    
    # Figure setup using GridSpec
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2]) # Left columns for the robot animatio, right one for the position plots


    # --- 1. Robot Sublots (Left column, they occupies 2 rows) ---
    ax_ani = fig.add_subplot(gs[:, 0])

    # --- INTELLIGENT TITLE MANAGEMENT ---
    # Dictionary mapping task_id to titles
    titles_map = {
        1: ("TASK 1", r"Gymnast Robot Animation for Trajectory Generation (I)"),
        2: ("TASK 2", r"Gymnast Robot Animation for Trajectory Generation (II)"),
        3: ("TASK 3", r"Tracking via LQR (III)"),
        4: ("TASK 4", r"Tracking via MPC (IV)")
    }

    # If task_id is in the dictionary, set the titles, otherwise use a default
    if task_id in titles_map:
        supt, subt = titles_map[task_id]
        fig.suptitle(supt, fontsize=16, fontweight='bold')
        ax_ani.set_title(subt, fontweight='bold')
    else:
        fig.suptitle("Gymnast Robot", fontsize=16, fontweight='bold')
        ax_ani.set_title("Animation", fontweight='bold')

    ax_ani.set_xlim(-L_tot, L_tot)
    ax_ani.set_ylim(-L_tot, L_tot)
    ax_ani.set_aspect('equal')
    ax_ani.grid(True, alpha=0.3)
    ax_ani.set_xlabel("X [m]")
    ax_ani.set_ylabel("Y [m]")

    # Graphical elements for Robot
    line_arm, = ax_ani.plot([], [], 'o-', lw=4, color='tab:blue', markersize=8, label='Robot')
    trace_line, = ax_ani.plot([], [], '-', lw=1, color='tab:red', alpha=0.5) # It creates the trace line behind
    
    # --- 2. Subplot Theta 1 (UP-DX) ---
    ax_th1 = fig.add_subplot(gs[0, 1])
    ax_th1.plot(time, np.degrees(xx[0, :]), 'b-', lw=2, label=r'Optimal $\theta_1$')
    if xx_ref is not None:
        ax_th1.plot(time, np.degrees(xx_ref[0, :]), 'k--', alpha=0.5, label='Ref')
    ax_th1.set_ylabel(r'$\theta_1$ [deg]')
    ax_th1.set_title(r"Angles Evolution", fontweight = 'bold')
    ax_th1.grid(True, linestyle='--')
    ax_th1.legend(loc='upper right')
    
    # Cursor (red dots)
    point_th1, = ax_th1.plot([], [], 'ro', zorder=5)
    
    # --- 3. Subplot Theta 2 (DX-DOWN) ---
    ax_th2 = fig.add_subplot(gs[1, 1], sharex=ax_th1)
    ax_th2.plot(time, np.degrees(xx[1, :]), 'r-', lw=2, label=r'Optimal $\theta_2$')
    if xx_ref is not None:
        ax_th2.plot(time, np.degrees(xx_ref[1, :]), 'k--', alpha=0.5, label='Ref')
    ax_th2.set_ylabel(r'$\theta_2$ [deg]')
    ax_th2.set_xlabel("Time [s]")
    ax_th2.grid(True, linestyle='--')
    ax_th2.legend(loc='upper right')
    
    point_th2, = ax_th2.plot([], [], 'ro', zorder=5)
    
    # Time text
    time_text = ax_ani.text(0.05, 0.95, '', transform=ax_ani.transAxes, fontsize=12)
    
    # Update for Animation 
    history_x, history_y = [], []
    
    # Skip frames in order to make the animation faster if dt is very small
    skip = max(1, int(0.05 / dt)) # Target ~20fps rendering
    
    def update(frame):
        idx = frame * skip
        if idx >= len(time): idx = len(time) - 1
        
        # Current state
        th1 = xx[0, idx]
        th2 = xx[1, idx]
        
        # Direct Kinematic
        x0, y0 = 0, 0
        x1 = l1 * np.sin(th1)
        y1 = -l1 * np.cos(th1)
        x2 = x1 + l2 * np.sin(th1 + th2)
        y2 = y1 - l2 * np.cos(th1 + th2)
        
        # Updating Robot
        line_arm.set_data([x0, x1, x2], [y0, y1, y2])
        
        # Updating trace
        history_x.append(x2)
        history_y.append(y2)
        trace_line.set_data(history_x, history_y)
        
        # Updating cursors
        t_curr = time[idx]
        point_th1.set_data([t_curr], [np.degrees(th1)])
        point_th2.set_data([t_curr], [np.degrees(th2)])
        
        time_text.set_text(f"Time: {t_curr:.2f} s")
        
        return line_arm, trace_line, point_th1, point_th2, time_text

    # Creating Animation
    n_frames = len(time) // skip
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=30, blit=True)

    return ani