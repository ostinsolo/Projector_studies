import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Training pairs: (camera image, projector source)
    TRAIN_PAIRS = [
        ("./real_data/1_image_distorted_origin.jpeg", "./real_data/1_pro.png"),
        ("./real_data/2_image_distorted_origin.jpeg", "./real_data/2_pro.png"),
        ("./real_data/3_image_distorted_origin.jpeg", "./real_data/3_pro.png"),
    ]

    MASK_EXTRACT_PATH = "./real_data/red_distorted_origin.jpeg"
    OUTPUT_DIR = "./neural_results_exp"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Camera and projector native resolutions
    H_CAM, W_CAM = 1440, 1920
    H_PROJ, W_PROJ = 1080, 1920
    # Downsampled training resolution
    TRAIN_SCALE = 0.25
    H_TRAIN = int(H_PROJ * TRAIN_SCALE)
    W_TRAIN = int(W_PROJ * TRAIN_SCALE)

    ITERS = 2500
    LR = 0.001
    num_freqs = 0  # positional encoding frequency count

    # Loss weights
    W_PHOTO_FWD = 10.0
    W_PHOTO_BWD = 10.0
    W_CYCLE = 10
    W_SMOOTH = 0.1
    W_MASK = 5
    W_GRADIENT = 0.3
    W_L1 = 0.7

    # Pretrained model paths (set to None to train from scratch)
    PRETRAINED_C2P = None
    PRETRAINED_P2C = None

# ==========================================
# 2. Network
# ==========================================
class PositionalEncoding(nn.Module):

    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        # x: [B, ..., 2] -> [B, ..., 2 + 2*2*num_freqs]
        res = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            res.append(torch.sin(x * freq * math.pi))
            res.append(torch.cos(x * freq * math.pi))
        return torch.cat(res, dim=-1)


class CoordinateNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, num_freqs=1):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=num_freqs)
        encoded_dim = in_dim + in_dim * 2 * num_freqs

        self.net = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, out_dim)
        )
        # Zero-init last layer so initial delta is zero
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, coords):
        coords_encoded = self.pe(coords)
        delta = self.net(coords_encoded)
        return coords + delta

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel kernels for edge detection
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).view(1, 1, 3, 3)

        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def get_gradient_map(self, img):
        """
        Input:  [B, 3, H, W] or [B, 1, H, W]
        Output: [B, 1, H, W] normalised gradient magnitude
        """
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # Per-sample min-max normalisation
        B, C, H, W = grad_mag.shape
        flat = grad_mag.view(B, -1)
        min_v = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_v = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        grad_norm = (grad_mag - min_v) / (max_v - min_v + 1e-6)
        return grad_norm

    def get_binary_gradient_map(self, img, threshold=0.1):
        """
        Input:  [B, 3, H, W] or [B, 1, H, W]
        Output: [B, 1, H, W] binary gradient map
        """
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        grad_x = F.conv2d(gray, self.kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.kernel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        binary_grad = (grad_mag > threshold).float()
        return binary_grad

    def forward(self, pred, target, mask=None):
        """L1 distance between gradient maps of pred and target."""
        g_pred = self.get_gradient_map(pred)
        g_target = self.get_gradient_map(target)

        # Occasionally dump a debug image
        if torch.rand(1) < 0.001:
           debug_img = (g_pred[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
           cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "debug_gradient_pred.png"), debug_img)
           print("[Debug] Saved gradient visualization: debug_gradient_pred.png")

        diff = torch.abs(g_pred - g_target)
        if mask is not None:
            if mask.shape[1] == 3:
                mask = mask[:, 0:1, :, :]
            return (diff * mask).sum() / (mask.sum() + 1e-6)
        else:
            return diff.mean()

# ==========================================
# 3. Utilities
# ==========================================
def preprocess_and_crop(img_path):
    """Crop _origin images to the active region, return the cropped path."""
    if "origin" in img_path:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img_cropped = img[160:160 + 1728, :3072]
        new_path = img_path.replace("_origin", "")
        cv2.imwrite(new_path, img_cropped)
        return new_path
    return img_path

def load_image_tensor(path, size=None, crop=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop:
        img = img[160:160 + 1728, :3072]
        cv2.imwrite("cropped_image.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if size: img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(Config.DEVICE) / 255.0

def create_mask(img_path, size=None):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    # Use the red channel for foreground extraction
    red_channel = img[:, :, 2]
    _, binary = cv2.threshold(red_channel, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(Config.DEVICE) / 255.0

    cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "camera_mask.png"), mask)

    return mask_tensor, mask

def get_base_grid(h, w):
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    return torch.stack((x, y), dim=-1).unsqueeze(0).to(Config.DEVICE)

def get_smooth_loss(grid):
    # grid: [1, H, W, 2]
    dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    return torch.mean(dx**2) + torch.mean(dy**2)

# ==========================================
# 4. Training
# ==========================================
class NeuralWarper:
    def __init__(self, MODE="train"):
        self.MODE = MODE
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        print("\n[Init] Preprocessing images (cropping)...")
        self.processed_train_pairs = []
        for cam_origin, proj_path in Config.TRAIN_PAIRS:
            cam_cropped = preprocess_and_crop(cam_origin)
            self.processed_train_pairs.append((cam_cropped, proj_path))

        self.test_cam_path = self.processed_train_pairs[2][0]
        self.test_proj_path = self.processed_train_pairs[2][1]

        # Geometric networks (with optional positional encoding)
        self.net_c2p = CoordinateNet(in_dim=2, out_dim=2, num_freqs=Config.num_freqs).to(Config.DEVICE)
        self.net_p2c = CoordinateNet(in_dim=2, out_dim=2, num_freqs=Config.num_freqs).to(Config.DEVICE)

        # Load pretrained weights if provided
        if Config.PRETRAINED_C2P and os.path.exists(Config.PRETRAINED_C2P):
            self.net_c2p.load_state_dict(torch.load(Config.PRETRAINED_C2P, map_location=Config.DEVICE))
            print(f"[Model] Loaded C2P: {Config.PRETRAINED_C2P}")

        if Config.PRETRAINED_P2C and os.path.exists(Config.PRETRAINED_P2C):
            self.net_p2c.load_state_dict(torch.load(Config.PRETRAINED_P2C, map_location=Config.DEVICE))
            print(f"[Model] Loaded P2C: {Config.PRETRAINED_P2C}")

        self.gradient_loss = GradientLoss().to(Config.DEVICE)

        # Load mask from the red-channel extract image
        mask_extract = cv2.imread(Config.MASK_EXTRACT_PATH)
        mask_extract = mask_extract[160:160 + 1728, :3072]
        cv2.imwrite("./real_data/mask_red.jpeg", mask_extract)
        print("Saved cropped camera mask: ./real_data/mask_red.jpeg")

        if self.MODE == "train":
            self.mask_tensor, self.mask_pic = create_mask("./real_data/mask_red.jpeg", (Config.W_TRAIN, Config.H_TRAIN))
        else:
            self.mask_tensor, self.mask_pic = create_mask("./real_data/mask_red.jpeg", (Config.W_CAM, Config.H_CAM))

        self.loss_history = {
            'total': [],
            'fwd': [],
            'bwd': [],
            'cycle': [],
            'smooth': [],
            'mask': []
        }

    def train(self):
        print(f"\n[Training — real capture mode] Target resolution: {Config.W_TRAIN}x{Config.H_TRAIN}")
        print(f"[Data] {len(self.processed_train_pairs)} image pairs loaded for training")

        # Preload all training data
        self.training_data_cache = []

        for idx, (cam_path, proj_path) in enumerate(self.processed_train_pairs):
            print(f"  -> Loading pair {idx+1}: Cam={os.path.basename(cam_path)}, Proj={os.path.basename(proj_path)}")

            t_img_cam = load_image_tensor(cam_path, (Config.W_TRAIN, Config.H_TRAIN))
            t_img_proj = load_image_tensor(proj_path, (Config.W_TRAIN, Config.H_TRAIN))

            self.training_data_cache.append({
                "img_cam": t_img_cam,
                "img_proj": t_img_proj
            })

            # Save gradient maps for the first pair as a debug reference
            if idx == 0:
                grad_cam = self.gradient_loss.get_gradient_map(t_img_cam)
                grad_proj = self.gradient_loss.get_gradient_map(t_img_proj)
                grad_cam_img = (grad_cam[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                grad_proj_img = (grad_proj[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(Config.OUTPUT_DIR + "/gradient_cam_ref.png", grad_cam_img)
                cv2.imwrite(Config.OUTPUT_DIR + "/gradient_proj_ref.png", grad_proj_img)

        # Base coordinate grids (shared across all pairs)
        base_grid_cam = get_base_grid(Config.H_TRAIN, Config.W_TRAIN)
        base_grid_proj = get_base_grid(Config.H_TRAIN, Config.W_TRAIN)

        optimizer = optim.Adam(
            list(self.net_c2p.parameters()) +
            list(self.net_p2c.parameters()),
            lr=Config.LR
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.ITERS)
        # Projector mask — full white (assumes full-frame projection)
        mask_proj = torch.ones((1, 1, Config.H_TRAIN, Config.W_TRAIN)).to(Config.DEVICE)

        pure_train_time = 0.0
        for i in range(Config.ITERS):
            step_start_time = time.time()
            optimizer.zero_grad()

            # Stochastic training — pick a random pair each iteration
            rand_idx = np.random.randint(len(self.training_data_cache))
            batch_data = self.training_data_cache[rand_idx]

            img_cam = batch_data["img_cam"]
            img_proj = batch_data["img_proj"]

            # 1. Projector -> Camera (P2C): coordinate mapping is p2c, image warp is c2p
            c_coords_pred = self.net_p2c(base_grid_proj)
            warped_cam_real = F.grid_sample(img_cam, c_coords_pred, align_corners=True, padding_mode='border')

            loss_grad_bwd = self.gradient_loss(warped_cam_real, img_proj)
            loss_l1_bwd = torch.mean(torch.abs(warped_cam_real - img_proj))
            loss_bwd = Config.W_GRADIENT * loss_grad_bwd + Config.W_L1 * loss_l1_bwd

            # P2C mask consistency
            fake_mask_proj = F.grid_sample(self.mask_tensor, c_coords_pred, align_corners=True, padding_mode='zeros')
            loss_mask_p2c = F.mse_loss(fake_mask_proj, mask_proj)

            # 2. Camera -> Projector (C2P)
            p_coords_pred = self.net_c2p(base_grid_cam)
            warped_proj_shaded = F.grid_sample(img_proj, p_coords_pred, align_corners=True)

            loss_grad_fwd = self.gradient_loss(warped_proj_shaded, img_cam, self.mask_tensor)
            loss_l1_fwd = torch.mean(torch.abs(warped_proj_shaded - img_cam) * self.mask_tensor)
            loss_fwd = Config.W_GRADIENT * loss_grad_fwd + Config.W_L1 * loss_l1_fwd

            fake_mask_cam = F.grid_sample(mask_proj, p_coords_pred, align_corners=True, padding_mode='zeros')
            loss_mask_c2p = F.mse_loss(fake_mask_cam, self.mask_tensor)

            # Cycle consistency + smoothness
            p_recon = self.net_c2p(c_coords_pred)
            loss_cycle_proj = torch.mean(torch.abs(p_recon - base_grid_proj))
            c_recon = self.net_p2c(p_coords_pred)
            loss_cycle_cam = torch.mean(torch.abs(c_recon - base_grid_cam) * self.mask_tensor.permute(0,2,3,1))
            loss_cycle = loss_cycle_cam + loss_cycle_proj

            loss_sm = get_smooth_loss(p_coords_pred) + get_smooth_loss(c_coords_pred)

            total_loss = (Config.W_PHOTO_FWD * loss_fwd) + \
                         (Config.W_PHOTO_BWD * loss_bwd) + \
                         (Config.W_CYCLE * loss_cycle) + \
                         (Config.W_SMOOTH * loss_sm) + \
                         (Config.W_MASK * (loss_mask_c2p + loss_mask_p2c))

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            step_end_time = time.time()
            pure_train_time += (step_end_time - step_start_time)

            self.loss_history['total'].append(total_loss.item())
            self.loss_history['fwd'].append(loss_fwd.item())
            self.loss_history['bwd'].append(loss_bwd.item())
            self.loss_history['cycle'].append(loss_cycle.item())
            self.loss_history['smooth'].append(loss_sm.item())
            self.loss_history['mask'].append((loss_mask_c2p + loss_mask_p2c).item())

            if i % 50 == 0:
                print(f"Iter {i} [Img {rand_idx}] | Loss: {total_loss.item():.4f} | Fwd: {loss_fwd.item():.4f} | Mask: {(loss_mask_c2p + loss_mask_p2c).item():.4f} | Bwd: {loss_bwd.item():.4f} | Cycle: {loss_cycle.item():.4f} | Smooth: {loss_sm.item():.4f}")

            # Save intermediate results every 1000 steps
            if (i + 1) % 1000 == 0:
                self.save_hd_results(iteration=i+1)

        print(f"\n[Timing] Pure training time (excl. eval & save): {pure_train_time:.2f}s")

        self.mask_tensor, self.mask_pic = create_mask("./real_data/mask_red.jpeg", (Config.W_CAM, Config.H_CAM))
        self.save_hd_results()
        self.visualize_projector_mask()
        self.save_loss_to_csv()

    def save_loss_to_csv(self):
        """Export loss history to CSV for external plotting."""
        import csv
        save_path = os.path.join(Config.OUTPUT_DIR, "loss_metrics.csv")

        keys = list(self.loss_history.keys())
        rows = zip(*[self.loss_history[k] for k in keys])

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)

        print(f"Loss data exported to: {save_path}")

    def plot_loss_curve(self):
        """
        Loss curve plot — tall portrait aspect, exponential smoothing,
        Times New Roman, no grid, per-component dashed lines.
        """
        if not self.loss_history['total']: return

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "Ubuntu Condensed"]
        plt.rcParams["font.size"] = 16

        plt.figure(figsize=(7, 12))

        def get_smoothed(data, weight=0.95):
            last = data[0]
            smoothed = []
            for val in data:
                new_val = last * weight + (1 - weight) * val
                smoothed.append(new_val)
                last = new_val
            return smoothed

        configs = [
            ('fwd',    'Photo Forward',  '#1f77b4', '--'),
            ('bwd',    'Photo Backward', '#ff7f0e', '-.'),
            ('cycle',  'Cycle Consist',  '#2ca02c', ':'),
            ('smooth', 'Geometric Reg',  '#d62728', (0, (3, 5, 1, 5))),
            ('mask',   'Mask Loss',      '#9467bd', (0, (1, 1))),
        ]

        x = range(len(self.loss_history['total']))

        for key, label, color, ls in configs:
            if key in self.loss_history:
                raw = self.loss_history[key]
                plt.plot(x, get_smoothed(raw), label=label, color=color, linestyle=ls, linewidth=1.5)

        total_s = get_smoothed(self.loss_history['total'])
        plt.plot(x, total_s, label='Total Loss', color='black', linestyle='-', linewidth=3)

        plt.yscale('log')
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss (Log Scale)')
        plt.grid(False)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.legend(loc='upper right', frameon=True, fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(Config.OUTPUT_DIR, "loss_portrait_smooth.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss curve saved to: {save_path}")

    @torch.no_grad()
    def save_hd_results(self, iteration=None, img_path=None):
        print(f"\n[HD Inference] Camera: {Config.W_CAM}x{Config.H_CAM}, Projector: {Config.W_PROJ}x{Config.H_PROJ}")

        suffix = f".iter{iteration}" if iteration is not None else ""
        torch.save(self.net_c2p.state_dict(), os.path.join(Config.OUTPUT_DIR, f"net_c2p{suffix}.pth"))
        torch.save(self.net_p2c.state_dict(), os.path.join(Config.OUTPUT_DIR, f"net_p2c{suffix}.pth"))

        # Coordinate grids at native resolutions
        grid_proj_hd = get_base_grid(Config.H_PROJ, Config.W_PROJ)
        grid_cam_hd = get_base_grid(Config.H_CAM, Config.W_CAM)

        # Chunked inference to avoid OOM
        _pure_infer_start = time.time()
        chunk_size = 200000
        with torch.no_grad():
            # Camera -> Projector
            grid_cam_flat = grid_cam_hd.view(-1, 2)
            p_coords_flat = torch.zeros_like(grid_cam_flat)
            for i in range(0, grid_cam_flat.shape[0], chunk_size):
                p_coords_flat[i:i+chunk_size] = self.net_c2p(grid_cam_flat[i:i+chunk_size])
            p_coords_hd = p_coords_flat.view(1, Config.H_CAM, Config.W_CAM, 2)

            # Projector -> Camera
            grid_proj_flat = grid_proj_hd.view(-1, 2)
            c_coords_flat = torch.zeros_like(grid_proj_flat)
            for i in range(0, grid_proj_flat.shape[0], chunk_size):
                c_coords_flat[i:i+chunk_size] = self.net_p2c(grid_proj_flat[i:i+chunk_size])
            c_coords_hd = c_coords_flat.view(1, Config.H_PROJ, Config.W_PROJ, 2)
        _pure_infer_net_end = time.time()

        # Load full-res images
        if img_path is None:
            t_cam = load_image_tensor(self.test_cam_path, (Config.W_CAM, Config.H_CAM))
            t_proj = load_image_tensor(self.test_proj_path, (Config.W_PROJ, Config.H_PROJ))
            img_without_distortion = self.generate_cam_without_distortion(self.test_proj_path)
        else:
            t_proj = load_image_tensor(img_path, (Config.W_PROJ, Config.H_PROJ))
            img_without_distortion = self.generate_cam_without_distortion(img_path)

        _pure_infer_grid_start = time.time()

        if img_path is None:
            fake_proj = F.grid_sample(t_cam, c_coords_hd, align_corners=True)

        pre_warped = F.grid_sample(img_without_distortion, c_coords_hd, align_corners=True)
        fake_cam = F.grid_sample(t_proj, p_coords_hd, align_corners=True)
        _pure_infer_grid_end = time.time()

        if iteration is None:
            print(f"\n[Timing] Network inference: {_pure_infer_net_end - _pure_infer_start:.4f}s")
            print(f"[Timing] Grid sample: {_pure_infer_grid_end - _pure_infer_grid_start:.4f}s")
            print(f"[Timing] Total inference (net + grid): {(_pure_infer_net_end - _pure_infer_start) + (_pure_infer_grid_end - _pure_infer_grid_start):.4f}s\n")

        if img_path is None:
            self.save_tensor(fake_proj, f"fake_proj{suffix}.png")
            self.save_tensor(pre_warped, f"pre_warped{suffix}.png")
            self.save_tensor(fake_cam, f"fake_cam{suffix}.png")
            fake_proj_grad = self.gradient_loss.get_gradient_map(fake_proj)
            fake_proj_grad_img = (fake_proj_grad[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(Config.OUTPUT_DIR, f"fake_proj_gradient{suffix}.png"), fake_proj_grad_img)
        else:
            self.save_tensor(pre_warped, f"pre_warped{suffix}.png")
            self.save_tensor(fake_cam, f"fake_cam{suffix}.png")

        print("All HD results and models saved.")

        # Quantitative metrics (final results only)
        if iteration is None:
            fake_cam_path = os.path.join(Config.OUTPUT_DIR, f"fake_cam.png")
            cam_ground_truth_path = self.test_cam_path
            SSIM_forward = self.calculate_ssim(fake_cam_path, cam_ground_truth_path, Config.OUTPUT_DIR + "/camera_mask.png")
            print(f"Camera recon SSIM: {SSIM_forward:.4f}")
            SSIM_Masked, PSNR_Masked, RMSE_Masked = self.calculate_metrics(fake_cam_path, cam_ground_truth_path, Config.OUTPUT_DIR + "/camera_mask.png")
            print(f"Camera recon (masked) -- SSIM: {SSIM_Masked:.4f}, PSNR: {PSNR_Masked:.4f}, RMSE: {RMSE_Masked:.4f}")

            fake_proj_path = os.path.join(Config.OUTPUT_DIR, f"fake_proj.png")
            proj_ground_truth_path = self.test_proj_path
            SSIM_inverted = self.calculate_ssim(fake_proj_path, proj_ground_truth_path)
            print(f"Projector recon SSIM: {SSIM_inverted:.4f}")
            SSIM_simulated_proj, PSNR_simulated_proj, RMSE_simulated_proj = self.calculate_metrics(fake_proj_path, proj_ground_truth_path)
            print(f"Projector recon -- SSIM: {SSIM_simulated_proj:.4f}, PSNR: {PSNR_simulated_proj:.4f}, RMSE: {RMSE_simulated_proj:.4f}")

    def calculate_ssim(self, img1_path, img2_path, mask_path=None):
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        import cv2

        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(img2_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if mask_path is None:
            ssim_value = ssim(img1, img2, channel_axis=-1, data_range=img2.max() - img2.min())
            return ssim_value

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask.shape[:2] != img1.shape[:2]:
            print("Resizing mask to match image dimensions.")
            mask = cv2.resize(mask, (img1.shape[1], img1.shape[0]))

        if img1.shape != img2.shape:
            print("Resizing img2 to match img1 dimensions.")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        mask_bool = mask > 0

        if np.sum(mask_bool) < 100:
            print("Mask region too small, computing full-image SSIM.")
            ssim_value = ssim(img1, img2, channel_axis=-1, data_range=img2.max() - img2.min())
            return ssim_value

        ssim_map = ssim(img1, img2, channel_axis=-1,
                    data_range=img2.max() - img2.min(),
                    win_size=7,
                    gradient=False,
                    gaussian_weights=True,
                    full=True)[1]

        ssim_masked = np.mean(ssim_map[mask_bool])
        return ssim_masked

    def calculate_metrics(self, img1_path, img2_path, mask_path=None):
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        import cv2
        import os

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            print(f"[Error] Could not read images: {img1_path} or {img2_path}")
            return 0, 0, 0

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if img1_rgb.shape != img2_rgb.shape:
            img2_rgb = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != img1_rgb.shape[:2]:
                mask = cv2.resize(mask, (img1_rgb.shape[1], img1_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0
        else:
            mask_bool = np.ones(img1_rgb.shape[:2], dtype=bool)

        if np.sum(mask_bool) < 10:
            return 0, 0, 0

        # SSIM map, then average over mask
        ssim_res = ssim(img1_rgb, img2_rgb, channel_axis=-1,
                        data_range=255, win_size=7, full=True)
        ssim_map = ssim_res[1]
        ssim_masked = np.mean(ssim_map[mask_bool])

        # RMSE & PSNR over masked pixels
        pixels1 = img1_rgb[mask_bool].astype(np.float64)
        pixels2 = img2_rgb[mask_bool].astype(np.float64)

        mse_val = np.mean((pixels1 - pixels2) ** 2)
        rmse_val = np.sqrt(mse_val)

        if mse_val < 1e-10:
            psnr_val = 100.0
        else:
            psnr_val = 10 * np.log10((255.0 ** 2) / mse_val)

        return ssim_masked, psnr_val, rmse_val

    def generate_cam_without_distortion(self, img_without_distortion_path):
        img_without_distortion = cv2.imread(img_without_distortion_path)
        img_without_distortion = cv2.cvtColor(img_without_distortion, cv2.COLOR_BGR2RGB)

        x, y, w_rect, h_rect = cv2.boundingRect(self.mask_pic)

        if w_rect > 0 and h_rect > 0:
            print(f"Mask bounding rect: x={x}, y={y}, w={w_rect}, h={h_rect}")

            img_resized = cv2.resize(img_without_distortion, (w_rect, h_rect), interpolation=cv2.INTER_LANCZOS4)

            new_img_without_distortion = np.zeros((Config.H_CAM, Config.W_CAM, 3), dtype=np.uint8)
            new_img_without_distortion[y:y+h_rect, x:x+w_rect] = img_resized

            debug_bgr = cv2.cvtColor(new_img_without_distortion, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "new_img_without_distortion.png"), debug_bgr)

            new_img_without_distortion = torch.from_numpy(new_img_without_distortion.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE) / 255.0
            return new_img_without_distortion
        else:
            print("Warning: empty mask, cannot crop/scale.")

    @torch.no_grad()
    def visualize_projector_mask(self, iteration=None):
        print("\n[Visualization] Generating projector-space mask (mask_on_proj)...")

        mask_cam, _ = create_mask("./real_data/mask_red.jpeg", (Config.W_CAM, Config.H_CAM))

        grid_proj_hd = get_base_grid(Config.H_PROJ, Config.W_PROJ)

        # Chunked p2c inference
        chunk_size = 200000
        grid_proj_flat = grid_proj_hd.view(-1, 2)
        c_coords_flat = torch.zeros_like(grid_proj_flat)
        with torch.no_grad():
            for i in range(0, grid_proj_flat.shape[0], chunk_size):
                c_coords_flat[i:i+chunk_size] = self.net_p2c(grid_proj_flat[i:i+chunk_size])
        c_coords_hd = c_coords_flat.view(1, Config.H_PROJ, Config.W_PROJ, 2)

        # Sampling: a white projector image, warped through c_coords,
        # tells us which projector pixels land on valid camera content.
        mask_on_proj = F.grid_sample(mask_cam, c_coords_hd, align_corners=True)

        mask_np = (mask_on_proj.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Also transform a white projector mask to the camera plane
        white_mask_proj_tensor = torch.ones((1, 1, Config.H_PROJ, Config.W_PROJ), device=Config.DEVICE)

        grid_cam_hd = get_base_grid(Config.H_CAM, Config.W_CAM).to(Config.DEVICE)

        # Chunked c2p inference
        grid_cam_flat = grid_cam_hd.view(-1, 2)
        p_coords_flat = torch.zeros_like(grid_cam_flat)
        with torch.no_grad():
            for i in range(0, grid_cam_flat.shape[0], chunk_size):
                p_coords_flat[i:i+chunk_size] = self.net_c2p(grid_cam_flat[i:i+chunk_size])
        p_coords_hd = p_coords_flat.view(1, Config.H_CAM, Config.W_CAM, 2)

        transformed_white_to_cam = F.grid_sample(white_mask_proj_tensor, p_coords_hd, align_corners=True)
        transformed_np = (transformed_white_to_cam.squeeze().cpu().numpy() * 255).astype(np.uint8)
        save_path_trans = os.path.join(Config.OUTPUT_DIR, f"mask_cam_fake{f'.iter{iteration}' if iteration is not None else ''}.png")
        cv2.imwrite(save_path_trans, transformed_np)
        print(f"Projector white mask transformed to camera plane: {save_path_trans}")

        save_path = os.path.join(Config.OUTPUT_DIR, "Final_Mask_on_Projector_Space" + (f".iter{iteration}" if iteration is not None else "") + ".png")
        cv2.imwrite(save_path, mask_np)
        print(f"Projector mask saved to: {save_path}")

        # Overlay mask on projector image for visual check
        img_proj = cv2.imread(self.test_proj_path)
        img_proj = cv2.resize(img_proj, (Config.W_PROJ, Config.H_PROJ))

        mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0
        mask_color[:, :, 1] = 0  # keep only red channel

        overlay = cv2.addWeighted(img_proj, 0.7, mask_color, 0.3, 0)
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "Debug_Mask_Overlay_on_Proj" + (f".iter{iteration}" if iteration is not None else "") + ".png"), overlay)

    @torch.no_grad()
    def visualize_displacement(self):
        """Plot displacement field with quiver arrows over a magnitude heatmap."""

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]

        print("\n[Visualization] Generating pixel displacement analysis...")
        self.net_c2p.eval()
        self.net_p2c.eval()

        configs = [
            (self.net_c2p, Config.H_CAM, Config.W_CAM, "cam_to_proj"),
            (self.net_p2c, Config.H_PROJ, Config.W_PROJ, "proj_to_cam")
        ]

        for model, h, w, name in configs:
            grid = get_base_grid(h, w)
            output = model(grid)
            delta_fwd_norm = (grid - output).squeeze(0).cpu().numpy()

            # Convert normalised coords to pixels
            dx_pixel = delta_fwd_norm[..., 0] * (w / 2.0)
            dy_pixel = delta_fwd_norm[..., 1] * (h / 2.0)
            mag_pixel = np.sqrt(dx_pixel**2 + dy_pixel**2)

            plt.figure(figsize=(14, 10))

            im = plt.imshow(mag_pixel, cmap='viridis', origin='upper', vmin=0, vmax=255)
            cbar = plt.colorbar(im, shrink=0.6, pad=0.04)

            ticks = [0, 50, 100, 150, 200, 255]
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels([str(t) for t in ticks], family='Times New Roman')
            cbar.set_label('Displacement Magnitude (Pixels)', fontsize=14, family='Times New Roman')

            step = 64 if w > 1000 else 32
            y_pos, x_pos = np.mgrid[step//2:h:step, step//2:w:step]
            u = dx_pixel[step//2:h:step, step//2:w:step]
            v = dy_pixel[step//2:h:step, step//2:w:step]

            plt.quiver(x_pos, y_pos, u, v, color='white', alpha=0.8,
                    angles='xy', scale_units='xy', scale=1,
                    width=0.0015, headwidth=4)

            plt.axis('off')

            save_path = os.path.join(Config.OUTPUT_DIR, f"shrink_viridis_{name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Displacement field saved to: {save_path}")

    def generate_checkerboard(self, h, w, s=80):
        cb = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, s):
            for x in range(0, w, s):
                if ((x//s)+(y//s)) % 2 == 0: cb[y:y+s, x:x+s] = 255
        return torch.from_numpy(cb.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE)/255.0

    def save_tensor(self, t, name):
        img = t.squeeze(0).permute(1,2,0).cpu().numpy() * 255
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, name), img)

    def load_models(self):
        c2p_path = os.path.join(Config.OUTPUT_DIR, "net_c2p.pth")
        p2c_path = os.path.join(Config.OUTPUT_DIR, "net_p2c.pth")

        if not os.path.exists(c2p_path) or not os.path.exists(p2c_path):
            print(f"[Error] Model files not found:\n  {c2p_path}\n  {p2c_path}")
            return False

        print(f"[Load] Loading weights from: {Config.OUTPUT_DIR}")
        try:
            self.net_c2p.load_state_dict(torch.load(c2p_path, map_location=Config.DEVICE))
            self.net_p2c.load_state_dict(torch.load(p2c_path, map_location=Config.DEVICE))
            self.net_c2p.eval()
            self.net_p2c.eval()
            print("Models loaded successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to load models: {e}")
            return False

    def run_inference(self, img_path=None):
        if self.load_models():
            start_time = time.time()
            self.save_hd_results(img_path=img_path)
            self.visualize_projector_mask()
            print(f"Inference done. Elapsed: {time.time()-start_time:.2f}s")
        else:
            print("Cannot run inference — train first to generate model files.")

if __name__ == "__main__":
    import sys

    MODE = sys.argv[1] if len(sys.argv) > 1 else "inference_custom"

    warper = NeuralWarper(MODE=MODE)

    if MODE == "train":
        warper.train()
    elif MODE == "inference":
        warper.run_inference(img_path=None)
    elif MODE == "inference_custom":
        custom_path = sys.argv[2] if len(sys.argv) > 2 else "./real_data/test.jpg"
        warper.run_inference(img_path=custom_path)
    else:
        print("Unknown mode. Use 'train' or 'inference'.")


# --- procam-calibrate adaptation patch ---
import os as _os
_os.chdir(r"/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/calibration/cspr_work")
# Force relative paths to work directory
Config.TRAIN_PAIRS = [
    ("./real_data/1_image_distorted_origin.jpeg", "./real_data/1_pro.png"),
    ("./real_data/2_image_distorted_origin.jpeg", "./real_data/2_pro.png"),
    ("./real_data/3_image_distorted_origin.jpeg", "./real_data/3_pro.png"),
]
Config.MASK_EXTRACT_PATH = "./real_data/red_distorted_origin.jpeg"
Config.OUTPUT_DIR = "./neural_results_exp"
Config.H_CAM, Config.W_CAM = 1440, 1920
Config.H_PROJ, Config.W_PROJ = 1080, 1920
Config.H_TRAIN = int(Config.H_PROJ * Config.TRAIN_SCALE)
Config.W_TRAIN = int(Config.W_PROJ * Config.TRAIN_SCALE)
Config.ITERS = 2500

def preprocess_and_crop(img_path):
    """Adaptation: no CSPR hardcoded crop; use capture as-is."""
    import cv2 as _cv
    img = _cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    out = img_path.replace("_origin.jpeg", ".jpeg").replace("_origin.jpg", ".jpg")
    if out == img_path:
        out = img_path + ".cropped.jpeg"
    _cv.imwrite(out, img)
    print(f"[adapt] using uncropped capture -> {out} size={img.shape[1]}x{img.shape[0]}")
    return out

