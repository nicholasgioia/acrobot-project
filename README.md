# Optimal Control of a Planar Gymnast Robot (Acrobot) 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CasADi](https://img.shields.io/badge/CasADi-Optimization-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-Scientific_Computing-lightgrey.svg)
![Control Theory](https://img.shields.io/badge/Control_Theory-LQR_%7C_MPC-success.svg)

## 1. 📖 Overview
This project implements an optimal control framework for a planar gymnast robot (**Acrobot**), modeled as an underactuated double pendulum actuated only at the hip (the second joint). 

The repository includes dynamics modeling, offline trajectory optimization via the **Regularized Newton's Method**, and online tracking using **Linear Quadratic Regulator (LQR)** and **Linear Time-Varying Model Predictive Control (LTV-MPC)**.

---

## 2. 🗂️ Repository Structure

### A. Core & Configuration
* `main.py`: Entry point executing Tasks 0-4 based on configuration.
* `config.py`: Defines physical parameters (mass, length, inertia), simulation settings (dt, horizon), and cost matrices.

### B. Dynamics & Modeling
* `dynamics.py`: Implements symbolic equations of motion using **CasADi**, including Inertia, Coriolis, Gravity, and Friction terms. Provides RK4 integration and linearization.
* `equilibrium.py`: Numerically computes equilibrium states and holding torques using root-finding algorithms.

### C. Optimization & Control
* `newtons_method.py`: Implements the Regularized Newton's method for iterative trajectory optimization.
* `LQR_affine.py`: Solves the Finite-Horizon LQR problem for LTV systems with affine terms, used for the backward pass.
* `mpc_solver.py`: Implements Linear Time-Varying MPC using CasADi to solve constrained optimization problems over a prediction horizon.
* `armijo.py`: Implements Armijo backtracking line search for optimal step-size selection.
* `cost.py`: Defines stage and terminal cost functions (Quadratic) and their gradients.
* `trajectory.py`: Generates initial guess trajectories using Step inputs or 5th-order Polynomials.

### D. Utilities
* `animation.py`: Renders robot dynamics and plots state evolution in real-time.
* `plot_task.py`: Generates performance plots (states, inputs, descent).

---

## 3. ⚙️ Requirements
To run the code, ensure you have the following Python libraries installed:
* `numpy`
* `scipy`
* `matplotlib`
* `casadi`
* `control`

You can install the dependencies via pip:
```bash
pip install numpy scipy matplotlib casadi control
```

## 👨‍💻 Authors
* Nicholas Gioia
* Tommaso Scagliarini
* Giorgio Soricetti

