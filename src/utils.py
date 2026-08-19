import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_quantum_state(model, device):
    """
    Plots the Real, Imaginary, and Probability Density of the Wavefunction psi(x, t)
    """
    model.eval()
    x = torch.linspace(-5, 5, 200).reshape(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)  # Snapshot at t = 0
    
    with torch.no_grad():
        u, v = model(x, t)
        
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    prob_density = u_np**2 + v_np**2
    x_np = x.cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_np, u_np, label=r'Real Part $u(x, t=0)$', color='blue')
    plt.plot(x_np, v_np, label=r'Imaginary Part $v(x, t=0)$', color='orange', linestyle='--')
    plt.plot(x_np, prob_density, label=r'Probability Density $|\psi|^2$', color='green', linewidth=2)
    
    plt.title('Quantum Wavefunction Snapshot via PINN')
    plt.xlabel('Position (x)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_wavefunction.png', dpi=300)
    print("Graph saved successfully as 'quantum_wavefunction.png'!")