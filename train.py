import torch
import torch.optim as optim
from src.schrodinger_pinn import QuantumPINN, schrodinger_residual
from src.utils import plot_quantum_state

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = QuantumPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Generate domain coordinates (x in [-5, 5], t in [0, 1])
    N_f = 2000
    x_f = (torch.rand(N_f, 1) * 10.0 - 5.0).to(device)
    t_f = torch.rand(N_f, 1).to(device)

    print("Starting Quantum PINN Training...")
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        loss = schrodinger_residual(model, x_f, t_f)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}/1000 | Physics Loss: {loss.item():.6f}")

    return model, device

if __name__ == "__main__":
    trained_model, current_device = train()
    plot_quantum_state(trained_model, current_device)