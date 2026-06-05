import torch
import random


# -------------------------
# Gaussian Noise
# -------------------------
def gaussian_noise(x, sigma=25):
    sigma = sigma / 255.0
    noise = torch.randn_like(x) * sigma
    return x + noise


# -------------------------
# Poisson Noise
# -------------------------
def poisson_noise(x):
    x_clamped = torch.clamp(x, 0.0, 1.0)
    noise = torch.poisson(x_clamped * 255.0) / 255.0
    return noise


# -------------------------
# Salt & Pepper Noise
# -------------------------
def salt_pepper_noise(x, prob=0.01):
    noisy = x.clone()
    rand = torch.rand_like(x)

    noisy[rand < prob / 2] = 0.0
    noisy[rand > 1 - prob / 2] = 1.0

    return noisy


# -------------------------
# MIXED NOISE (MAIN FUNCTION)
# -------------------------
def add_mixed_noise(x, sigma_range=(0, 75)):
    noise_type = random.choice(["gaussian", "poisson", "sp"])

    if noise_type == "gaussian":
        sigma = random.uniform(*sigma_range)
        return gaussian_noise(x, sigma)

    elif noise_type == "poisson":
        return poisson_noise(x)

    elif noise_type == "sp":
        return salt_pepper_noise(x)

    # fallback (safe)
    return gaussian_noise(x, random.uniform(*sigma_range))