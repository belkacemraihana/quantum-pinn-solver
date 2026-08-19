import torch
import torch.nn as nn

class QuantumPINN(nn.Module):
    """
    Physics-Informed Neural Network for solving 1D Time-Dependent Schrödinger Equation.
    Predicts Real (u) and Imaginary (v) components of the wavefunction psi(x, t).
    """
    def __init__(self, layers=[2, 100, 100, 100, 100, 2]):
        super(QuantumPINN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()
        
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(self.activation)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        
        self.network = nn.Sequential(*layer_list)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        outputs = self.network(inputs)
        u = outputs[:, 0:1]  # Real part
        v = outputs[:, 1:2]  # Imaginary part
        return u, v

def schrodinger_residual(model, x, t, hbar=1.0, m=1.0):
    """
    Computes the quantum physics residual using Automatic Differentiation:
    i*hbar * d(psi)/dt + (hbar^2 / 2m) * d2(psi)/dx2 - V(x)*psi = 0
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u, v = model(x, t)
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    
    V = 0.5 * (x ** 2)  # Harmonic Oscillator potential
    
    # Real and Imaginary Residuals
    f_u = -hbar * v_t + (hbar**2 / (2 * m)) * u_xx - V * u
    f_v =  hbar * u_t + (hbar**2 / (2 * m)) * v_xx - V * v
    
    loss_f = torch.mean(f_u**2) + torch.mean(f_v**2)
    return loss_f