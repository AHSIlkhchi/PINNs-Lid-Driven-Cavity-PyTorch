# PINNs-Lid-Driven-Cavity-PyTorch
A Physics-Informed Neural Network (PINN) approach to solving the lid-driven cavity problem using PyTorch. This implementation integrates both data-driven and physics-constrained techniques for improved accuracy in fluid simulations.

# Overview
This project models the lid-driven cavity problem using PINNs, where the neural network learns the velocity field (u, v) while implicitly solving for pressure (p). The solution is time-dependent, meaning various time sections can be analyzed instead of a steady-state solution.

# Network Architecture
Input: (x, y, t) spatial and temporal coordinates.
Output: (u, v, p) velocity and pressure at each point.
Layers: 8 layers
Activation: Tanh for all layers except the last (default linear activation).
Neurons per layer:
(3, 30)
(30, 60)
(60, 60)
(60, 60)
(60, 60)
(60, 30)
(30, 30)
(30, 3)
Network Construction: Implemented using PyTorch’s sequential method.

# Domain and Boundary Conditions
Domain: Generated using meshgrid over (x, y, t).
Boundary Conditions:
No-slip walls: Zero velocity on all walls except the moving lid.
Lid velocity: u = 2 m/s at the top wall.
Initial Conditions: Zero velocity across the entire domain at t = 0.

# Governing Equations
The model enforces physical constraints through the Navier-Stokes equations:
Momentum Equation (X-direction):

$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

Momentum Equation (Y-direction):

$$ \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) $$

Continuity Equation:

$$ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 $$

These equations are incorporated into the loss function, ensuring the network learns physically consistent solutions.

# Training Details
Loss Function: Mean Squared Error (MSELoss).
Optimization Strategy:
First, trained using Adam optimizer for 2000 iterations.
Then refined using L-BFGS optimizer for improved convergence.
Loss Calculation: Combines data loss (boundary conditions) and PDE loss (Navier-Stokes equations).

# Post-Processing
Loss Visualization: Total loss curve over iterations.
Flow Visualization: Contour plots of velocity magnitude with streamlines.

# How to Run
Clone the repository:
git clone https://github.com/YOUR_USERNAME/PINNs-Lid-Driven-Cavity

Install dependencies:
pip install torch numpy matplotlib seaborn

Run the training script:
python pinn_cavity.py

# Results
The trained PINN successfully captures the shear-driven flow dynamics and vortices, demonstrating smooth velocity contours and physically reasonable streamlines.
