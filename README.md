# Quantum Hamilton-Jacobi PINN Solver ⚛️🧠

An advanced **Physics-Informed Neural Network (PINN)** framework designed to solve the **Time-Dependent Schrödinger Equation** for quantum dynamic systems using PyTorch and Automatic Differentiation.

## 📌 Features
- **Quantum State Prediction**: Approximates complex wavefunction $\psi(x, t) = u(x, t) + i v(x, t)$.
- **Physical Loss Constraints**: Enforces zero residual on the Quantum Schrödinger Field dynamics.
- **Deep Learning for Quantum Physics**: Implements continuous spatio-temporal boundary conditions without labeled data.

## 📐 Governing Physics
$$\hbar \frac{\partial \psi}{\partial t} = \hat{H} \psi = \left( -\frac{\hbar^2}{2m} \nabla^2 + V(x) \right) \psi$$

## 📁 Repository Structure
```text
quantum-pinn-solver/
├── src/
│   ├── __init__.py
│   └── schrodinger_pinn.py   # PINN architecture & residual loss
├── train.py                   # Training pipeline
└── README.md                  # Documentation