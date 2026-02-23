import numpy as np
from config import Config

cfg = Config()

nx = cfg.nx
nu = cfg.nu

def stagecost(xx, uu, xx_ref, uu_ref, QQt, RRt):
  """
    Stage-cost
 
    Quadratic cost function representing the "penalty" at each time step:
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)
 
    Args:
      - xx: Current state vector
      - uu: Current input vector
      - xx_ref: Target state reference
      - uu_ref: Target input reference
      - QQt: State weight matrix (penalizes tracking error)
      - RRt: Control weight matrix (penalizes control effort)
 
    Returns:
      - ll: Scalar cost value at the current step
      - lx: Gradient of the cost with respect to state x
      - lu: Gradient of the cost with respect to input u
  """

  # Reshape arrays to column vectors (N, 1) for consistent matrix multiplication
  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  return ll.squeeze(), lx, lu

def termcost(xT, xT_ref, QQT):
  """
    Terminal-cost
 
    Quadratic cost function applied only to the final state of the horizon:
    l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)
 
    Args:
      - xT: Final state vector at time T
      - xT_ref: Final state reference
      - QQT: Terminal weight matrix (usually higher than stage cost Q)
 
    Returns:
      - llT: Scalar terminal cost
      - lTx: Gradient of the terminal cost wrt final state x
  """
  # Compute terminal cost
  llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref)
  # Compute gradient of terminal cost
  lTx = QQT@(xT - xT_ref)


  return llT.squeeze(), lTx