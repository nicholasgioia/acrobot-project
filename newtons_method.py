import numpy as np
import cost as cst
import LQR_affine as lqr
import armijo as arm
import control as ctrl
from dynamics import AcrobotDynamics
from config import Config

cfg = Config()
dyn = AcrobotDynamics()

def newton_method (dynamics, xx, uu, xx_ref, uu_ref, x0, max_iters, task_number, armijo_plot = True, armijo_plot_number = 2):
    """
    REGULARIZED Newton's method for optimal control of a pendubot.
 
    Parameters
    ----------
    xx : array, shape (nx, TT, max_iters+1)        Decision variable states.
    uu : array, shape (nu, TT, max_iters+1)        Decision variable inputs.
    xx_ref : array, shape (nx, TT)                 Reference curve states.
    uu_ref : array, shape (nu, TT)                 Reference curve inputs.
    x0 : array, shape (nx,)                        Initial condition.
    max_iters : int                                Maximum number of iterations.
    task_number : int                              Task number, either 1 or 2.
    armijo_plot : bool, optional                   Flag to plot Armijo rule
    armijo_plot_number : int, optional             Number of iterations to plot Armijo rule
 
    Returns
    -------
    xx : array, shape (nx, TT, max_iters+1)        Decision variable states at each iteration.
    uu : array, shape (nu, TT, max_iters+1)        Decision variable inputs at each iteration.
    descent : array, shape (max_iters+1,)          Descent at each iteration.
    J : array, shape (max_iters+1,)                Cost at each iteration.
    kk : int                                       Number of iterations.
    """

    cfg = dynamics.cfg
    nx = cfg.nx
    nu = cfg.nu
    TT = uu.shape[1]

    # Armijo parameters
    c = cfg.ARMIJO_C
    beta = cfg.ARMIJO_BETA
    armijo_maxiters = cfg.ARMIJO_MAX_IT   
    stepsize_0 = cfg.ARMIJO_STEPSIZE_0         
    term_cond = cfg.ARMIJO_TERM_COND

     # Import the cost matrices 
    if task_number == 1:
        Qt, Rt, QT = cfg.Qt_task1, cfg.Rt_task1, cfg.QT_task1

    elif task_number == 2:
        Qt, Rt, QT = cfg.Qt_task2, cfg.Rt_task2, cfg.QT_task2
    else: 
        print("\n\n\nInvalid task number, stopping the algorithm...")
        quit()
    
    # Computing QQT as the solution of the discrete algebraic Riccati equation
    # Linearize around the reference trajectory
    AA_ref,BB_ref = dynamics.get_linearization_np(xx_ref[:,-1], uu_ref[:,-1])

    # Solve DARE to get terminal cost matrix
    QT = ctrl.dare(AA_ref,BB_ref, Qt, Rt)[0] 


    # Linearization
    A = np.zeros((nx, nx, TT))
    B = np.zeros((nx, nu, TT))

    # Derivatives/Gradients of the cost (affine terms of the cost)
    q = np.zeros((nx, TT))
    r = np.zeros((nu, TT))

    # Affine dynamic term (not used since it's equal to zero)
    cc = np.zeros((nx,TT))

    # Initial conditions
    xx0 = np.zeros((nx,))

    # Cost matrices (regularized - so with only cost Hessian)
    Qtilda = np.zeros((nx, nx, TT))
    Rtilda = np.zeros((nu, nu, TT))
    Stilda = np.zeros((nu, nx, TT))

    # lambda for the co-state equation
    lmbd = np.zeros((nx, TT, max_iters+1))    

    # Cost and descent direction 
    dJ = np.zeros((nu,TT, max_iters+1))       
    J = np.zeros(max_iters+1)                 
    descent = np.zeros(max_iters+1)           
    descent_arm = np.zeros(max_iters+1)       

    # Decision variables
    deltax = np.zeros((nx, TT, max_iters+1)) 
    deltau = np.zeros((nu, TT, max_iters+1)) 


    for kk in range(max_iters):
        J[kk] = 0

        # Forward pass to compute cost and linearization
        for tt in range(TT-1):
            temp_cost= cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
            J[kk] += temp_cost
            
            # Linearization at (x_k, u_k)
            A[:,:,tt], B[:,:,tt] = dynamics.get_linearization_np(xx[:,tt,kk], uu[:,tt,kk])
            
            Qtilda[:,:,tt] = Qt
            Rtilda[:,:,tt] = Rt

        # Terminal cost
        term_cost, qT = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QT)
        QTilda = QT
        J[kk] += term_cost

        # Descent direction calculation
        lmbd_temp = qT # Terminal condition for the costate
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
        
        # Solve costate equation and gradient of J wrt u backwards in time
        for tt in reversed(range(TT-1)): 
            qt, rt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[1:]

            lmbd_temp = A[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + qt        # Costate equation
            dJ_temp  =  B[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + rt        # Gradient of J wrt u 
            
            q[:,tt] = qt.squeeze()
            r[:,tt] = rt.squeeze()
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()

        # Solve the Affine LQR problem through REGULARIZATION
        deltax[:,:,kk], deltau[:,:,kk], KK, sigma, _ = lqr.ltv_LQR_affine(A, B, Qtilda, Rtilda, Stilda, QTilda, TT, xx0, q, r, qT.squeeze())

        for tt in reversed(range(TT)): 
            descent[kk] += abs(dJ[:,tt,kk].T@deltau[:,tt,kk])
            descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk] 

        # Select stepsize using Armijo rule
        stepsize = arm.select_stepsize(dynamics,stepsize_0, armijo_maxiters, c, beta, sigma, xx_ref, uu_ref, x0, uu[:, :, kk], xx[:, :, kk], KK, J[kk], descent_arm[kk], kk, Qt, Rt, QT, armijo_plot, armijo_plot_number)

        # Update the current solution
        xx_temp = np.zeros((nx,TT))
        uu_temp = np.zeros((nu,TT))

        xx_temp[:,0] = x0

        # Forward integration, closed-loop with robustified step 2
        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt]@(xx_temp[:,tt]-xx[:,tt,kk]) + stepsize*sigma[:,tt]
            xx_temp[:,tt+1] = dynamics.rk4_step(xx_temp[:,tt], uu_temp[:,tt])

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition
        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}\n'.format(kk,descent[kk], J[kk]))

        if descent[kk] <= term_cond:
            max_iters = kk
            break
        
    return xx[:, :, :kk+2], uu[:, :, :kk+2], descent, J, kk