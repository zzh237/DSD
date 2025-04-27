import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import cdist
import copy
import sys
sys.path.append("./networks")
from training.toy_example_networks import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ========= Config =========
class Config:
    n_samples       = 1000
    noise_sigma     = 0.1
    latent_dim      = 32
    hidden_dim      = 64
    low_rank_dim    = 1
    batch_size      = 128
    pretrain_epochs = 300
    distill_epochs  = 800
    lr              = 1e-4
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed            = 42
    plot_size       = (10,5)
    data_dim        = 2

cfg = Config()
torch.manual_seed(cfg.seed)

# ========= Data =========
def create_spiral(n):
    t = torch.linspace(0, 4*math.pi, n)
    r = 1 + t/(4*math.pi)
    x = torch.stack([r*torch.cos(t), r*torch.sin(t)], dim=1)
    return x

clean_data = create_spiral(cfg.n_samples).to(cfg.device)



# ========= Ambient Tweedie Loss =========
def ambient_tweedie_loss(model, y, x_t, t, sigma, sigma_t):
    # 2) Compute the coefficients for the target
    coef1 = ((sigma_t**2 - sigma**2) / sigma_t**2).view(-1,1)
    coef2 = ( sigma**2                / sigma_t**2).view(-1,1)

    # 3) The input to the model is the actual time step t
    pred   = model(x_t, t)         # [B,2]
    target = coef1 * pred + coef2 * x_t  # [B,2]
    return ((target - y)**2).mean()

def score_calculation(model, x_t, t, sigma, sigma_t):
    # 2) Compute the coefficients for the target
    coef1 = ((sigma_t**2 - sigma**2) / sigma_t**2).view(-1,1)
    coef2 = ( sigma**2                / sigma_t**2).view(-1,1)

    # 3) The input to the model is the actual time step t
    pred   = model(x_t, t)         # [B,2]
    target = coef1 * pred + coef2 * x_t  # [B,2]
    return (target - x_t) / (sigma_t**2 - sigma**2)

# ========= Phase I: Pretrain Teacher =========
def pretrain_teacher(y_data, sigma, sigma_t_min, t_sigma):
    teacher = DiffusionMLP().to(cfg.device)
    opt     = optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-5)
    dl      = DataLoader(TensorDataset(y_data), batch_size=cfg.batch_size, shuffle=True)
    sched   = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(cfg.pretrain_epochs):
        running_loss = 0.0
        for (y,) in dl:
            y = y.to(cfg.device)
            B = y.size(0)

            t, sigma_t = schedule_t(t_sigma, sigma_t_min, sigma, B, y.device)

            # Generate noise
            eps = torch.randn_like(y)          # [B,2]
            delta = torch.sqrt((sigma_t**2 - sigma**2).clamp(min=1e-6))
            delta = delta.view(-1,1)            # [B,1]
            # 1) Construct x_t
            x_t = y + delta * eps               # [B,2]

            loss = ambient_tweedie_loss(teacher, y, x_t, t, sigma, sigma_t.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dl)
        sched.step(avg_loss)
        if epoch % 100 == 0:
            print(f"[Pretrain {epoch:4d}] loss={loss.item():.4f}")
    return teacher

def schedule_t(t_sigma, sigma_t_min, sigma, B, device):
    t = t_sigma + (1.0 - t_sigma) * torch.rand(B, 1, device=device)
    sigma_t = sigma_t_min + t * (1.0 - sigma_t_min)
    sigma_t = torch.clamp(sigma_t, min=sigma)
    return t, sigma_t



# ========= Phase II: Distill =========
def train_dsd(generator, teacher, sigma, sigma_t_min, t_sigma):
    fake_diff = copy.deepcopy(teacher).to(cfg.device)  # copy from teacher, separate parameters
    # generator = copy.deepcopy(teacher).to(cfg.device)  # example: copying for generator, separate parameters
    optG = optim.Adam(generator.parameters(), lr=cfg.lr)
    optD = optim.Adam(fake_diff.parameters(), lr=cfg.lr)

    for epoch in range(cfg.distill_epochs):
        # =================================================================
        # Phase I: Update the virtual diffusion model f_ψ  # lines 7 to 9
        # =================================================================
        # Generate latent samples
        z = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
        # Sample time steps (line 9 in the algorithm)
        B = z.size(0)
        t, sigma_t = schedule_t(t_sigma, sigma_t_min, sigma, B, cfg.device)

        with torch.no_grad():
            x_g = generator(z, t)
        # x_g = generator(z)           # [B,2]
        # Add noise (line 8 in the algorithm)
        noise = torch.randn_like(x_g)
        tilde_y = x_g + sigma * noise  # this does not contain generator gradients
        x_t = x_g + sigma_t * noise
        # 1) Update fake diffusion model
        lossD = ambient_tweedie_loss(
            fake_diff,
            tilde_y.detach(),
            x_t,
            t,
            sigma,
            sigma_t.view(-1)
        )
        optD.zero_grad(); lossD.backward(); optD.step()
        
        # =================================================================
        # Phase II: Update the generator G_θ  # lines 10 to 12
        # =================================================================
        # Regenerate data to decouple gradients
        z = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
        # Sample new time steps
        t, sigma_t = schedule_t(t_sigma, sigma_t_min, sigma, B, cfg.device)
        x_g = generator(z, t)
        # Construct x_t = G(z) + σ_t * ε
        noise = torch.randn_like(x_g)

        tilde_y = x_g + sigma * noise  # includes generator gradients
        x_t = x_g + sigma_t * noise
        
        # Inner loop parameters for bilevel optimization
        max_inner_iters = 400
        tol = 5e-5

        prev_loss = float('inf')
        stop_iter = max_inner_iters - 1
        final_loss = None

        # Inner loop: update fake_diff until convergence
        for i in range(max_inner_iters):
            loss_inner = ((fake_diff(x_t.detach(), t) - x_g.detach())**2).mean()
            optD.zero_grad(); loss_inner.backward(); optD.step()

            curr_loss = loss_inner.item()
            # Check convergence and break if change below tolerance
            if abs(prev_loss - curr_loss) < tol:
                stop_iter = i
                final_loss = curr_loss
                break
            prev_loss = curr_loss

        # If never stopped early, set final_loss to the last computed loss
        if final_loss is None:
            final_loss = curr_loss

        # Compute score s_{σ,σ_t}(x_t) (Equation 11)
        with torch.no_grad():
            x0_hat = fake_diff(x_t, t)
        # x0_hat = teacher(x_t, t.view(-1)).detach()      # [B,2]
        
        lossG = ambient_tweedie_loss(fake_diff, tilde_y, x0_hat, t, sigma, sigma_t)   
        
        # Update generator parameters (line 12 in the algorithm)
        optG.zero_grad(); lossG.backward(); optG.step()

        if epoch % 100 == 0:
            print(f"[Distill {epoch:4d}] G_loss={lossG.item():.4f}, D_loss={lossD.item():.4f}")

    return generator



# def train_dsd(generator, teacher, sigma, sigma_t_min, t_sigma):
#     fake_diff = copy.deepcopy(teacher).to(cfg.device)  # copy from teacher, separate parameters
#     # generator = copy.deepcopy(teacher).to(cfg.device)  # copy from teacher, separate parameters
#     optG = optim.Adam(generator.parameters(), lr=cfg.lr)
#     optD = optim.Adam(fake_diff.parameters(), lr=cfg.lr)

#     for epoch in range(cfg.distill_epochs):
#         # =================================================================
#         # Phase I: Update the virtual diffusion model f_ψ  # lines 7 to 9
#         # =================================================================
#         # Generate latent samples
#         z = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
#         # Sample time steps (line 9 in the algorithm)
#         B = z.size(0)
#         t, sigma_t = schedule_t(t_sigma, sigma_t_min, sigma, B, cfg.device)

#         with torch.no_grad():
#             x_g = generator(z, t)
#         # x_g = generator(z)           # [B,2]
#         # Add noise (line 8 in the algorithm)
#         noise = torch.randn_like(x_g)
#         tilde_y = x_g + sigma * noise  # this does not contain generator gradients
#         x_t = x_g + sigma_t * noise
#         # 1) Update fake diffusion model
#         lossD = ambient_tweedie_loss(fake_diff, tilde_y.detach(), x_t, t, sigma, sigma_t.view(-1))
#         optD.zero_grad(); lossD.backward(); optD.step()

#         # =================================================================
#         # Phase II: Update the generator G_θ  # lines 10 to 12
#         # =================================================================
#         # Regenerate data to decouple gradients
#         z = torch.randn(cfg.batch_size, cfg.data_dim, device=cfg.device)
#         # Sample new time steps
#         t, sigma_t = schedule_t(t_sigma, sigma_t_min, sigma, B, cfg.device)
#         x_g = generator(z, t)
#         # Construct x_t = G(z) + σ_t * ε
#         noise = torch.randn_like(x_g)

#         tilde_y = x_g + sigma * noise  # contains generator gradients
#         x_t = x_g + sigma_t * noise

#         # Inner loop parameters for bilevel optimization
#         max_inner_iters = 400
#         tol = 5e-5

#         prev_loss = float('inf')
#         stop_iter = max_inner_iters - 1
#         final_loss = None

#         # Inner loop: update fake_diff until convergence
#         for i in range(max_inner_iters):
#             loss_inner = ((fake_diff(x_t.detach(), t) - x_g.detach())**2).mean()
#             optD.zero_grad()
#             loss_inner.backward()
#             optD.step()

#             curr_loss = loss_inner.item()
#             # Check convergence and break if change below tolerance
#             if abs(prev_loss - curr_loss) < tol:
#                 stop_iter = i
#                 final_loss = curr_loss
#                 break
#             prev_loss = curr_loss

#         # If never stopped early, set final_loss to the last computed loss
#         if final_loss is None:
#             final_loss = curr_loss

#         # Compute teacher score s_{σ,σ_t}(x_t) (Equation 11)
#         with torch.no_grad():
#             x0_hat = teacher(x_t, t)
#             # x0_hat = teacher(x_t, t.view(-1)).detach()      # [B,2]
#             coef1 = (sigma_t**2 - sigma**2) / sigma_t**2
#             coef2 = sigma**2 / sigma_t**2
#             # 1) Teacher-predicted posterior mean μ = coef1 · f_φ(x_t) + coef2 · x_t
#             mu = coef1 * x0_hat + coef2 * x_t
#             score_teacher = (mu - x_t) / (sigma_t**2 - sigma**2)
            
#         # Convert fake_diff learned posterior mean fψ*(x_t2) into student score
#         # Compute student score ∇log p_ψ(x_t) (Equation 13)
#         x0_pred = fake_diff(x_t, t)  # [B,2]
#         score_student = (x0_pred - x_t) / sigma_t**2  # [B,2]

#         # Distillation loss: align fψ*(x_t2)'s score with teacher's score (Equation 14)
#         lossG = ((score_teacher - score_student)**2).mean()

#         # Update generator parameters (line 12 in the algorithm)
#         optG.zero_grad(); lossG.backward(); optG.step()

#         if epoch % 100 == 0:
#             print(f"[Distill {epoch:4d}] G_loss={lossG.item():.4f}, D_loss={lossD.item():.4f}")

#     return generator

# ========= Utils & Main =========
def compute_w2(x, y):
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    if torch.is_tensor(y): y = y.detach().cpu().numpy()
    cost = cdist(x, y, 'sqeuclidean')
    return np.mean(np.min(cost, axis=1))

def visualize(noisy, student, title):
    plt.scatter(noisy[:,0], noisy[:,1], s=3, label="Teacher noisy")
    plt.scatter(student[:,0],student[:,1], s=3, label="Student")
    plt.title(title); plt.legend(); plt.axis('equal'); plt.xticks([]); plt.yticks([])

def get_t_sigma(device):
    # —— Randomly sample a t_sigma from (0,1] —— 
    # torch.rand returns values in [0,1), add a small offset to ensure it falls into (0,1]
    t_sigma = torch.rand(1, device=cfg.device)
    t_sigma = t_sigma.clamp(min=1e-6)
    return t_sigma 

if __name__ == "__main__":
    sigma      = cfg.noise_sigma
    f2         = (2 - math.sqrt(1+sigma**2))**2 * sigma**2 \
                 /(-4 + 4*math.sqrt(1+sigma**2) - sigma**2)
    sigma_t_ok = math.sqrt(max(f2,1e-6))
    print(f"Theoretical threshold σ_t² ≥ {sigma_t_ok**2:.4f}")
    print("Model     | σ_t label |  W2(teacher) |  W2(student)")
    print("-"*50)
    t_sigma = 0.1
    # Phase I
    teacher = pretrain_teacher(clean_data, sigma, sigma_t_ok, t_sigma)

    results = []
    for name, G_ctor in [
        ("LowRank", lambda: LowRankGenerator().to(cfg.device)),
        ("FreeMLP", lambda: FreeMLPGenerator().to(cfg.device)),
        ("Teacher", lambda: copy.deepcopy(teacher).to(cfg.device))
    ]:
        # for label, s_t in [("OK", sigma_t_ok), ("BAD", 0.2*sigma_t_ok)]:
        for label, s_t in [("OK", sigma_t_ok)]:        
            G        = G_ctor()
            G        = train_dsd(G, teacher, sigma, s_t, t_sigma)       
            with torch.no_grad():
                z       = torch.randn(cfg.n_samples, cfg.data_dim, device=cfg.device)
                # t, sigma_t = schedule_t(t_sigma, s_t, sigma, z.size(0), cfg.device)
                t = torch.zeros(z.size(0), device = cfg.device)  
                student = G(z,t).cpu().numpy()
                noisy   = (clean_data + sigma*torch.randn_like(clean_data)).cpu().numpy()

            w2T = compute_w2(noisy,   clean_data)
            w2S = compute_w2(student, clean_data)
            print(f"{name:8s}| {label:6s}|   {w2T:.4f}    |   {w2S:.4f}")
            results.append((name, label, noisy, student))

    # visualize all 4 cases
    n = len(results)
    cols = 2
    rows = math.ceil(n/cols)

    plt.figure(figsize=cfg.plot_size)
    for i, (m,label,noisy,stu) in enumerate(results,1):
        plt.subplot(rows, cols, i)
        visualize(noisy, stu, f"{m} | σ_t={label}")
    plt.tight_layout()
    plt.show()