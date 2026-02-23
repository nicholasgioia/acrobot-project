import numpy as np
from dynamics import AcrobotDynamics
from config import Config

cfg = Config()
dyn = AcrobotDynamics()


def ltv_LQR_affine(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
  Solves the Linear-Quadratic Regulator (LQR) problem for a Linear Time-Varying (LTV)
  system with an affine cost function using the discrete-time Riccati equation.
  
  This solver handles the standard quadratic terms and additional linear (affine)
  terms in the cost function, which are essential for tracking trajectories.
 
  Parameters:
    - AAin, BBin: State and Input dynamics matrices (can be constant or time-varying).
    - QQin, RRin, SSin: State, Input, and Cross-coupling cost matrices.
    - QQfin: Terminal state cost matrix.
    - TT: Time horizon (total steps).
    - x0: Initial state vector.
    - qqin, rrin, qqfin: Affine (linear) cost terms for state, input, and terminal state.
 
  Returns:
    - xxout: Optimized state trajectory over the horizon.
    - uuout: Optimized input (control) trajectory over the horizon.
    - KK: Sequence of optimal feedback gain matrices.
    - sigma: Sequence of optimal feedforward affine terms.
    - PP: Sequence of Riccati matrices (representing cost-to-go).
  """
	
  # Input Normalization
  # Ensures all input matrices are treated as 3D arrays (Matrix_dim1 x Matrix_dim2 x Time)
  try: 
    nx, lA = AAin.shape[1:]
  except:
    AAin = AAin[:,:,None]
    nx, lA = AAin.shape[1:]

  try:  
   nu, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    nu, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency for safety
  if nQ != nx:
    print("Matrix Q does not match number of states")
    exit()
  if nR !=nu:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs !=nx:
    print("Matrix S does not match number of states")
    exit()
  if nSi !=nu:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  augmented = False # Flag for augmented state

  if qqin is not None or rrin is not None or qqfin is not None:
    augmented = True
    print("Augmented term!")

  # Initialization
  KK = np.zeros((nu,nx, TT-1)) #K_t
  sigma = np.zeros((nu, TT-1)) #sigma_t
  PP = np.zeros((nx,nx, TT)) #P_t
  pp = np.zeros((nx, TT)) #p_t

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((nx, TT))
  uu = np.zeros((nu, TT))

  xx[:,0] = x0 #Initialization of the initial state
  
  #Terminal Cost (t=T)
  PP[:,:,-1] = QQf #P_T = Q_T
  pp[:,-1] = qqf #p_T = q_T
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)): # Solved backwards in time from T-1 to 0

    # Matrices of time t
    QQ_t = QQ[:,:,tt]
    RR_t = RR[:,:,tt]
    SS_t = SS[:,:,tt]
    AA_t = AA[:,:,tt]
    BB_t = BB[:,:,tt]

    # Matrices and vectors of time t+1
    PP_p = PP[:,:,tt+1] # P_t+1
    pp_p = pp[:,tt+1][:,None]   # p_t+1
    
    # Affine terms of time t
    qq_t = qq[:, tt][:,None]
    rr_t = rr[:, tt][:,None]
    
    # Decomposition of parts of Riccati equation computation
    # M_t = R_t + B_t^T * P_t+1 * B_t
    MMt = RR_t + BB_t.T @ PP_p @ BB_t
    # Inverse of M_t  M_t^(-1)
    MMt_inv = np.linalg.inv(MMt) 
    # prod_sigma = r_t + B_t^T p_t+1 (B_t^T P_t+1 c_t = 0  because c_t = 0)
    prod_sigma = rr_t + BB_t.T @ pp_p 
    # prod_k = S_t + B_t^T P_t+1 A_t old
    prod_k = SS_t + BB_t.T @ PP_p @ AA_t

    # 1. Optimal gain K_t and optimal feedforward term sigma_t
    # K_t = - M_t^{-1} * prod_k
    K_t_star = - MMt_inv @ prod_k
    # sigma_t = - M_t^{-1} * prod_sigma
    sigma_t_star = - MMt_inv @ prod_sigma

    # Store directly in the output arrays
    KK[:, :, tt] = K_t_star
    sigma[:, tt]  = sigma_t_star.squeeze()
    
    # 2. Riccati P_t (Discrete-time Riccati Equation - DRE)
    # P_t = Q_t + A_t^T P_t+1 A_t - K_t^T M_t K_t
    PPt = QQ_t + AA_t.T @ PP_p @ AA_t - K_t_star.T @ MMt @ K_t_star
    
    # 3. Affine vector p_t
    # p_t = q_t + A_t^T p_t+1 - K_t^T M_t sigma_t (A_t^T P_t+1 c_t = 0  because c_t = 0)
    ppt = qq_t + AA_t.T @ pp_p - K_t_star.T @ MMt @ sigma_t_star

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Forward Pass

  # Compute the new state-input trajectory (Forward Integration)
  for tt in range(TT - 1):
    
    # u_t = K_t * x_t + sigma_t
    uu[:, tt] = KK[:,:,tt] @ xx[:, tt] + sigma[:, tt]
    # x_{t+1} = A * x_t + B * u_t
    xx_p = AA[:,:,tt] @ xx[:,tt] + BB[:,:,tt] @ uu[:,tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return xxout, uuout, KK, sigma, PP

