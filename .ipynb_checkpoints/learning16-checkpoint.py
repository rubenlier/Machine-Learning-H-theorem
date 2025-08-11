import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from glob import glob

import torch.nn.functional as F


# --- Load all runs from the folder ---
file_paths = sorted(glob("velocities/run_*.pkl"))
all_velocitiesrun = []

for path in file_paths:
    with open(path, "rb") as f:
        run_data = pickle.load(f)
        all_velocitiesrun.append(np.array(run_data))  # shape: (frames, n_particles, 2)

all_runs = np.array(all_velocitiesrun)
n_runs, n_timesteps, n_particles, dims = all_runs.shape  # dims = 2 (vx, vy)

# --- DeepSets-style model ---
class DeepSetsHPotential(nn.Module):
    def __init__(self, particle_dim=2, phi_hidden=128, rho_hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(particle_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, 1)
        )

    def forward(self, x):
        # x: shape (B, N * 2)
        B = x.shape[0]
        x = x.view(B, n_particles, dims)        # → (B, N, 2)
        phi_x = self.phi(x)                     # → (B, N, D)
        summed = phi_x.sum(dim=1)               # → (B, D)
        return self.rho(summed).squeeze(-1)     # → (B,)

# --- Hyperparameters ---
n_hidden = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepSetsHPotential(particle_dim=2, phi_hidden=n_hidden, rho_hidden=n_hidden).to(device)
optimizer = optim.Adam(model.parameters(), lr=4e-4)
lambda_reg = 0.2

# --- Prepare flattened input pairs ---
all_pairs = []
for run in all_runs:
    for t in range(n_timesteps - 1):
        s_t = run[t]      # shape: (n_particles, 2)
        s_tp1 = run[t+1]
        all_pairs.append((s_t, s_tp1))

s_t_np = np.array([x for x, _ in all_pairs], dtype=np.float32)  # shape: (samples, n, 2)
s_tp1_np = np.array([y for _, y in all_pairs], dtype=np.float32)
s_t_all = torch.from_numpy(s_t_np).view(len(all_pairs), -1)     # flatten to (samples, n*2)
s_tp1_all = torch.from_numpy(s_tp1_np).view(len(all_pairs), -1)

# --- Training loop ---
batch_size = 512
n_epochs = 120
n_batches = len(all_pairs) // batch_size

for epoch in range(n_epochs):
    perm = torch.randperm(len(all_pairs))
    s_t_all = s_t_all[perm]
    s_tp1_all = s_tp1_all[perm]

    total_loss = 0
    total_loss0 = 0
    for i in range(n_batches):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        s_t = s_t_all[idx].to(device)
        s_tp1 = s_tp1_all[idx].to(device)

        h_t = model(s_t)
        h_tp1 = model(s_tp1)

        delta_h = h_tp1 - h_t
        
        
        # Compute dynamic leak factor
        negative_slope = 1 - 0.8 * epoch / (n_epochs - 1)
        
        # Apply LeakyReLU with dynamic slope
        leaky_dh = -F.leaky_relu(-delta_h, negative_slope=negative_slope)

        # Loss: penalized mean + optional regularization
        loss = -torch.mean(leaky_dh)+ lambda_reg * torch.mean(delta_h ** 2)
        loss0 = -torch.mean(delta_h)+ lambda_reg * torch.mean(delta_h ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss0  += loss0.item()

    with torch.no_grad():
            print(f"Epoch {epoch}: Loss: {total_loss / n_batches:.8f}, "f"⟨Δh⟩={delta_h.mean().item():.5f}, "f"Loss0: {total_loss0 / n_batches:.8f}")        

# --- Save model weights only ---
model_dir = os.path.join(os.getcwd(), "saved_model16")
os.makedirs(model_dir, exist_ok=True)
weights_path = os.path.join(model_dir, f"deepsets_h_model_weights.pt")

torch.save(model.state_dict(), weights_path)
print(f"Model weights saved to {weights_path}")
