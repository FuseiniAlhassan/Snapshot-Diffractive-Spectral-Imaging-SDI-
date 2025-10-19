import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader, Dataset
from matplotlib import animation
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import scipy.io as sio
import pandas as pd
import scipy.signal as signal
from scipy.sparse.linalg import cg
from PIL import Image

# Environment and settings

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

N = 64  # simulation grid
physical_size = 2e-3
dx = physical_size / N

num_bands = 9
wavelengths = torch.linspace(450e-9, 650e-9, num_bands, device=device).cpu().numpy()

z = 5e-3  # propagation distance
n_material = 1.5  # Refractive index for height map

batch_size = 32
epochs = 15
lr = 1e-3
use_hyperspec_data = False  # Set to False for now to debug with CIFAR-10 first
noise_level = {'shot': 1e-3, 'read': 1e-4}

# Dataset
class HyperspecDataset(Dataset):
    def __init__(self, root_dir='./data/hyperspec', transform=None, target_bands=num_bands):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.mat')] if os.path.exists(root_dir) else []
        self.target_bands = target_bands

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mat = sio.loadmat(os.path.join(self.root_dir, self.files[idx]))
        cube = mat.get('img', mat.get('hyper_image', np.random.rand(64, 64, 31)))  # Fallback
        cube = cube.astype(np.float32) / cube.max()

        # Resample spectral dimension if needed
        if cube.shape[-1] != self.target_bands:
            cube = np.interp(
                np.linspace(0, 1, self.target_bands),
                np.linspace(0, 1, cube.shape[-1]),
                cube, axis=-1
            )

        # Convert to tensor and ensure correct shape
        cube_tensor = torch.from_numpy(cube).float()
        if cube_tensor.dim() == 3:
            cube_tensor = cube_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Resize if needed
        if cube_tensor.shape[1] != N or cube_tensor.shape[2] != N:
            cube_tensor = F.interpolate(cube_tensor.unsqueeze(0), size=(N, N), mode='bilinear').squeeze(0)

        return cube_tensor  # Return only the cube, no label

# Use CIFAR-10 for simplicity during debugging
transform_cifar = T.Compose([T.ToTensor(), T.Resize((N, N))])

train_set_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
val_set_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Create data loaders with proper collate function
def cifar_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)  # (B, 3, H, W)
    # Convert RGB to pseudo-hyperspectral by repeating channels
    cubes = images.repeat(1, num_bands // 3 + (1 if num_bands % 3 > 0 else 0), 1, 1)[:, :num_bands, :, :]
    return cubes  # Return only cubes, no labels

train_subset = torch.utils.data.Subset(train_set_cifar, list(range(0, 2000)))
val_subset = torch.utils.data.Subset(val_set_cifar, list(range(0, 400)))

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=cifar_collate)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=cifar_collate)

# DOE: Trainable wavelength-dependent phase mask

base_phase = torch.nn.Parameter(torch.randn((N, N), device=device) * 2 * np.pi)
optimize_doe = True
optimizer_doe = torch.optim.Adam([base_phase], lr=0.05) if optimize_doe else None

lambda_ref = wavelengths[num_bands//2]

# Precompute transfer functions (Torch)
fx = torch.fft.fftfreq(N, d=dx, device=device)
FX, FY = torch.meshgrid(fx, fx, indexing='ij')
Hs = []
for wl in wavelengths:
    k = 2 * torch.pi / wl
    term = torch.clamp(1 - (wl * FX)**2 - (wl * FY)**2, min=0.0)
    H = torch.exp(1j * k * z * torch.sqrt(term))
    Hs.append(H)
Hs = torch.stack(Hs)  # (num_bands, N, N) complex

# Forward model: simulate sensor from spectral cube (differentiable)

def forward_sdi(cube, base_phase, add_noise=True):
    """cube: (B, num_bands, N, N)"""
    if len(cube.shape) == 3:
        cube = cube.unsqueeze(0)
    B = cube.shape[0]
    amp = torch.sqrt(cube.clamp(min=1e-6))
    phase_scale = torch.from_numpy(lambda_ref / wavelengths).float().to(device)
    phi = base_phase[None, None, :, :] * phase_scale[None, :, None, None]
    field = amp * torch.exp(1j * phi)

    # Propagate
    field_flat = field.view(B * num_bands, N, N)
    H_flat = Hs.repeat(B, 1, 1).view(B * num_bands, N, N)
    U = torch.fft.fft2(field_flat)
    U = U * H_flat
    u_out = torch.fft.ifft2(U)
    intensity = torch.abs(u_out) ** 2
    intensity = intensity.view(B, num_bands, N, N)
    sensor = intensity.sum(dim=1)
    sensor = sensor / (sensor.amax(dim=(1,2), keepdim=True) + 1e-12)

    # Add noise
    if add_noise:
        shot_noise = torch.poisson(sensor / noise_level['shot']) * noise_level['shot']
        read_noise = torch.randn_like(sensor) * noise_level['read']
        sensor = sensor + shot_noise + read_noise
        sensor = sensor.clamp(min=0)
    return sensor, cube

# Precompute PSFs for physics inversion
psfs = []
for i, wl in enumerate(wavelengths):
    impulse = torch.zeros((1, 1, N, N), device=device)
    impulse[0, 0, N//2, N//2] = 1.0
    with torch.no_grad():
        field = impulse * torch.exp(1j * base_phase.detach()[None, None, :, :] * (lambda_ref / wl))
        U = torch.fft.fft2(field.view(1, N, N))
        U = U * Hs[i:i+1]
        u_out = torch.fft.ifft2(U)
        intensity = torch.abs(u_out)**2
        psfs.append(intensity[0,0].cpu().numpy())

psfs = np.stack(psfs)
np.save('results/psfs.npy', psfs)

 # Physics-based inversion

def linear_forward_cube(X_cube):
    y = np.zeros((N, N), dtype=np.float32)
    for i in range(num_bands):
        y += signal.fftconvolve(X_cube[i], psfs[i], mode='same')
    return y

def tikhonov_solve(y, alpha=1e-2, maxiter=40):
    M = num_bands * N * N
    def matvec(v):
        V = v.reshape((num_bands, N, N))
        Av = linear_forward_cube(V)
        AtAv = np.zeros_like(V)
        for i in range(num_bands):
            AtAv[i] = signal.fftconvolve(Av, psfs[i][::-1, ::-1], mode='same')
        return (AtAv.flatten() + alpha * v)
    b = np.zeros((num_bands, N, N), dtype=np.float32)
    for i in range(num_bands):
        b[i] = signal.fftconvolve(y, psfs[i][::-1, ::-1], mode='same')
    b = b.flatten()
    x, info = cg(matvec, b, maxiter=maxiter)
    return x.reshape((num_bands, N, N)).clip(0,1)


# Advanced CNN: U-Net with Spectral Attention

class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SDINet(nn.Module):
    def __init__(self, in_ch=1, out_ch=num_bands, base=32):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base*2, 3, padding=1, stride=2)
        self.mid = nn.Conv2d(base*2, base*2, 3, padding=1)
        self.attn = SpectralAttention(base*2)
        self.up = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec = nn.Conv2d(base*2, base, 3, padding=1)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        m = self.attn(F.relu(self.mid(e2)))
        u = self.up(m)
        cat = torch.cat([u, e1], dim=1)
        d = F.relu(self.dec(cat))
        return torch.sigmoid(self.out(d))

model = SDINet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

if optimize_doe:
    opt.add_param_group({'params': [base_phase]})


# Training loop

sample_snapshots = []
metrics_log = {'epoch': [], 'psnr_avg': [], 'ssim_avg': []}

for epoch in range(epochs):
    model.train()
    if optimize_doe:
        base_phase.requires_grad_(True)
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

    for cubes in pbar:  # cubes is directly from collate_fn
        cubes = cubes.to(device)
        sensors, _ = forward_sdi(cubes, base_phase)
        inp = sensors.unsqueeze(1)
        pred = model(inp)
        loss = criterion(pred, cubes)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    print(f'Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}')

    # Validation
    model.eval()
    if optimize_doe:
        base_phase.requires_grad_(False)
    val_psnr, val_ssim = [], []

    with torch.no_grad():
        for cubes in val_loader:  # cubes is directly from collate_fn
            cubes = cubes.to(device)
            sensors, _ = forward_sdi(cubes, base_phase, add_noise=False)
            inp = sensors.unsqueeze(1)
            pred = model(inp).cpu().numpy()
            cubes_np = cubes.cpu().numpy()

            for b in range(pred.shape[0]):
                for band in range(num_bands):
                    val_psnr.append(psnr(cubes_np[b, band], pred[b, band], data_range=1.0))
                    val_ssim.append(ssim(cubes_np[b, band], pred[b, band], data_range=1.0))

    avg_psnr = np.mean(val_psnr)
    avg_ssim = np.mean(val_ssim)
    metrics_log['epoch'].append(epoch+1)
    metrics_log['psnr_avg'].append(avg_psnr)
    metrics_log['ssim_avg'].append(avg_ssim)
    print(f'Epoch {epoch+1} Val PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')

    # Snapshot for animation
    with torch.no_grad():
        sample_cube = cubes[0:1]  # First validation batch
        sample_sensor, _ = forward_sdi(sample_cube, base_phase)
        pred = model(sample_sensor.unsqueeze(1))[0].cpu().numpy()
        stacked = np.concatenate([
            sample_sensor[0].cpu().numpy()[None, ...],
            pred,
            sample_cube[0].cpu().numpy()
        ], axis=0)
        sample_snapshots.append(stacked)

# Save metrics
pd.DataFrame(metrics_log).to_csv('results/metrics.csv', index=False)

# Save optimized DOE and height map
def to_np(tensor):
    return tensor.detach().cpu().numpy()

optimized_phase = to_np(base_phase)
np.save('results/optimized_doe_phase.npy', optimized_phase)
height = (optimized_phase % (2*np.pi)) / (2*np.pi / lambda_ref * (n_material - 1)) * 1e6
plt.imshow(height, cmap='viridis')
plt.colorbar(label='Height (um)')
plt.savefig('results/height_map.png', dpi=300)
plt.close()

# Save model
torch.save(model.state_dict(), 'results/sdi_cnn.pth')

# Animations

fig = plt.figure(figsize=(10,6))
frames = []
for snap in sample_snapshots:
    grid = np.zeros((3*N, num_bands*N))
    sensor = snap[0]
    pred = snap[1:num_bands+1]
    true = snap[num_bands+1:]
    for i in range(num_bands):
        grid[0:N, i*N:(i+1)*N] = true[i]
        grid[N:2*N, i*N:(i+1)*N] = pred[i]
    grid[2*N:3*N, 0:N] = sensor
    im = plt.imshow(grid, cmap='gray', animated=True)
    plt.axis('off')
    frames.append([im])
ani = animation.ArtistAnimation(fig, frames, interval=800, blit=True)
ani.save('results/sdi_evolution.gif', writer='pillow')
plt.close()