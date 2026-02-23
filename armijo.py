import numpy as np
import matplotlib.pyplot as plt
import cost as cst



def select_stepsize(dynamics, stepsize_0, armijo_maxiters, cc, beta, sigma, xx_ref, uu_ref,  x0, uu, xx, KK, JJ, descent_arm, kk, Qt, Rt, QT, armijo_plot = False, armijo_plot_number = 3):

      """
      Computes the optimal stepsize (gamma) using Armijo's rule (Backtracking Line Search).
      This ensures that the cost function sufficiently decreases at each iteration.
 
      Parameters:
        - dynamics: Instance of AcrobotDynamics for forward simulation.
        - stepsize_0: The initial step size (usually 1.0).
        - armijo_maxiters: Max attempts to reduce step size.
        - cc: Sufficient decrease constant (0 < cc < 1).
        - beta: Reduction factor (0 < beta < 1) used to shrink the step size.
        - sigma: Feed-forward adjustment term from the LQR solver.
        - xx_ref, uu_ref: Reference state and input trajectories.
        - x0: Initial state of the robot.
        - uu, xx: Current input and state trajectories (from iteration k).
        - KK: Feedback gain matrix from the LQR solver.
        - JJ: Current total cost value at iteration k.
        - descent_arm: The expected cost reduction (gradient-based).
        - kk: Current iteration count of the main optimization loop.
        - Qt, Rt, QT: Cost weighting matrices.
        - armijo_plot: Boolean to enable visualization of the line search.
 
      Returns:
        - stepsize: The largest gamma that satisfies the Armijo condition.
      """

      TT = uu.shape[1]


      stepsizes = []  # list of stepsizes
      costs_armijo = [] # list of costs associated to the stepsizes

      stepsize = stepsize_0

      nx = xx_ref.shape[0]
      nu = uu_ref.shape[0]

      # Backtracking loop
      for ii in range(armijo_maxiters):

            # temp solution update
            xx_temp = np.zeros((nx,TT))
            uu_temp = np.zeros((nu,TT))

            xx_temp[:,0] = x0

            # Forward pass, simulate the system with new candidate stepsize
            for tt in range(TT-1):
                  # Robustified step 2 update, closed-loop forward integration
                  uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + stepsize * sigma[:,tt] 
                  # Get state at next time step
                  xx_temp[:,tt+1] = dynamics.rk4_step(xx_temp[:,tt], uu_temp[:,tt])

            JJ_temp = 0

            # Sum of the stage costs
            for tt in range(TT-1):
                  temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                  JJ_temp += temp_cost
            # Add terminal cost
            temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
            JJ_temp += temp_cost

            # save the stepsize
            stepsizes.append(stepsize)      

            # save the cost associated to the stepsize
            costs_armijo.append(JJ_temp)    

            # Armijo condition check
            if JJ_temp > JJ + cc*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize

            else:
                  #print(f'Armijo stepsize = {stepsize} at iteration k = {kk}')
                  break
            
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")

      print(f"Armijo at iteration {kk}: stepsize = {stepsize}")  
            
      ############################
      # Descent Armijo Plot
      ############################
      plt.rcParams["figure.figsize"] = (10,6)

      if armijo_plot and (kk < 3 or kk%10 == 0 or kk==armijo_plot_number or kk==7):

            # stepsizes for visualization (from 0 to stepsize_0 = 1, adding the armijo intermediate steps for better visualization)
            steps = list(np.linspace(0,stepsize_0,int(dynamics.cfg.ARMIJO_PLOT_RESOLUTION)))
            for iteration in range(ii):
                  arm_step_size = beta**(iteration+1)
                  if not (arm_step_size in steps): 
                        steps.append(beta**(iteration+1))
            steps.sort()
            steps = np.array(steps)

            costs = np.zeros(len(steps))

            for ii in range(len(steps)):
                  step = steps[ii]

                  # temp solution update
                  xx_temp = np.zeros((nx,TT))
                  uu_temp = np.zeros((nu,TT))

                  xx_temp[:,0] = x0
                  for tt in range(TT-1):
                        uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + step * sigma[:,tt]

                        xx_temp[:,tt+1] = dynamics.rk4_step(xx_temp[:,tt], uu_temp[:,tt])

                  # temp cost computation
                  JJ_temp = 0
                  for tt in range(TT-1):
                        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                        JJ_temp += temp_cost

                  temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
                  JJ_temp += temp_cost

                  costs[ii] = JJ_temp


            plt.figure(1)
            plt.clf()

      
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - \\gamma^k*d^k)$', linewidth = 2)
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - \\gamma^k*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - c*\\gamma^k\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)

            # plot the tested stepsize
            plt.scatter(stepsizes, costs_armijo, marker='*', s=100, zorder = 5) 

            plt.grid()
            plt.xlabel('$\\gamma^k$')
            plt.ylabel("$g(\\gamma^k)$")
            plt.legend()
            plt.title(f"Armijo rule at iteration k = {kk} | Cost J={JJ:.3e} | Descent={abs(descent_arm):.3e}")
            plt.draw()

            plt.show()

      return stepsize
