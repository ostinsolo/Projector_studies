import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
import time
import math

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Training pairs: (camera image, projector source)
    TRAIN_PAIRS = [
        ("./sim_data/1_image_distorted.png", "./sim_data/1_pro.png"),
        ("./sim_data/2_image_distorted.png", "./sim_data/2_pro.png"),
        ("./sim_data/3_image_distorted.png", "./sim_data/3_pro.png"),
    ]

    # Default pair used for inference / visualization
    CAM_IMAGE_PATH = TRAIN_PAIRS[2][0]
    REF_PATTERN_PATH = TRAIN_PAIRS[2][1]
    OUTPUT_DIR = "./neural_results_simulation"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Camera and projector native resolutions
    H_CAM, W_CAM = 1728, 3072
    H_PROJ, W_PROJ = 1080, 1920
    # Downsampled resolution for training
    TRAIN_SCALE = 0.25
    H_TRAIN = int(H_PROJ * TRAIN_SCALE)
    W_TRAIN = int(W_PROJ * TRAIN_SCALE)

    ITERS = 6000
    LR = 0.001

    # Loss weights
    W_PHOTO_FWD = 10.0
    W_PHOTO_BWD = 10.0
    W_CYCLE = 10
    W_SMOOTH = 0.1
    W_MASK = 5.0

# ==========================================
# 2. Network
# ==========================================

class CoordinateNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()

        # 2 -> 128 -> 256 -> 128 -> 2, with LayerNorm + LeakyReLU
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer — no norm/activation so delta can be unbounded
            nn.Linear(128, out_dim)
        )

        # Zero-init the last layer so initial offsets are zero
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, coords):
        # coords: [B, H, W, 2] or [B, N, 2]
        delta = self.net(coords)
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
def load_image_tensor(path, size=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size: img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(Config.DEVICE) / 255.0

def create_mask(img_path, size=None):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

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
    def __init__(self):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        # net_c2p: camera coords -> projector coords
        self.net_c2p = CoordinateNet(in_dim=2, out_dim=2).to(Config.DEVICE)
        # net_p2c: projector coords -> camera coords
        self.net_p2c = CoordinateNet(in_dim=2, out_dim=2).to(Config.DEVICE)
        self.gradient_loss = GradientLoss().to(Config.DEVICE)
        self.mask_cam, _ = create_mask(Config.CAM_IMAGE_PATH, (Config.W_TRAIN, Config.H_TRAIN))

    def train(self):
        print(f"\n[Training] Target resolution: {Config.W_TRAIN}x{Config.H_TRAIN}")
        print(f"[Data] {len(Config.TRAIN_PAIRS)} image pairs loaded for training")

        # Preload all training data
        self.training_data_cache = []

        for idx, (cam_path, proj_path) in enumerate(Config.TRAIN_PAIRS):
            print(f"  -> Loading pair {idx+1}: Cam={os.path.basename(cam_path)}, Proj={os.path.basename(proj_path)}")

            t_img_cam = load_image_tensor(cam_path, (Config.W_TRAIN, Config.H_TRAIN))
            t_img_proj = load_image_tensor(proj_path, (Config.W_TRAIN, Config.H_TRAIN))

            self.training_data_cache.append({
                "img_cam": t_img_cam,
                "img_proj": t_img_proj
            })

            # Save gradient maps for the second pair as a debug reference
            if idx == 1:
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
        # Projector mask — all white, assuming full-frame projection
        mask_proj = torch.ones((1, 1, Config.H_TRAIN, Config.W_TRAIN)).to(Config.DEVICE)

        for i in range(Config.ITERS):
            optimizer.zero_grad()

            # Randomly pick one pair per iteration (stochastic training)
            rand_idx = np.random.randint(len(self.training_data_cache))
            batch_data = self.training_data_cache[rand_idx]

            img_cam = batch_data["img_cam"]
            img_proj = batch_data["img_proj"]

            # 1. Projector -> Camera (P2C)
            c_coords_pred = self.net_p2c(base_grid_proj)
            warped_cam_real = F.grid_sample(img_cam, c_coords_pred, align_corners=True, padding_mode='border')

            loss_grad_bwd = self.gradient_loss(warped_cam_real, img_proj)
            loss_l1_bwd = torch.mean(torch.abs(warped_cam_real - img_proj))
            loss_bwd = 0.3 * loss_grad_bwd + 0.7 * loss_l1_bwd

            # P2C mask consistency
            fake_mask_proj = F.grid_sample(self.mask_cam, c_coords_pred, align_corners=True, padding_mode='zeros')
            loss_mask_p2c = F.mse_loss(fake_mask_proj, mask_proj)

            # 2. Camera -> Projector (C2P)
            p_coords_pred = self.net_c2p(base_grid_cam)
            warped_proj_shaded = F.grid_sample(img_proj, p_coords_pred, align_corners=True)

            loss_grad_fwd = self.gradient_loss(warped_proj_shaded, img_cam, self.mask_cam)
            loss_l1_fwd = torch.mean(torch.abs(warped_proj_shaded - img_cam) * self.mask_cam)
            loss_fwd = 0.3 * loss_grad_fwd + 0.7 * loss_l1_fwd

            fake_mask_cam = F.grid_sample(mask_proj, p_coords_pred, align_corners=True, padding_mode='zeros')
            loss_mask_c2p = F.mse_loss(fake_mask_cam, self.mask_cam)

            # Cycle consistency + smoothness
            p_recon = self.net_c2p(c_coords_pred)
            loss_cycle_proj = torch.mean(torch.abs(p_recon - base_grid_proj))
            c_recon = self.net_p2c(p_coords_pred)
            loss_cycle_cam = torch.mean(torch.abs(c_recon - base_grid_cam) * self.mask_cam.permute(0,2,3,1))
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

            if i % 50 == 0:
                print(f"Iter {i} [Img {rand_idx}] | Loss: {total_loss.item():.4f} | Fwd: {loss_fwd.item():.4f} | Mask: {(loss_mask_c2p + loss_mask_p2c).item():.4f} | Bwd: {loss_bwd.item():.4f} | Cycle: {loss_cycle.item():.4f} | Smooth: {loss_sm.item():.4f}")

        self.save_hd_results()
        self.visualize_projector_mask()

    @torch.no_grad()
    def save_hd_results(self, iteration=None, img_path=None):
        print(f"\n[HD Inference] Camera: {Config.W_CAM}x{Config.H_CAM}, Projector: {Config.W_PROJ}x{Config.H_PROJ}")

        suffix = f".iter{iteration}" if iteration is not None else ""
        torch.save(self.net_c2p.state_dict(), os.path.join(Config.OUTPUT_DIR, f"net_c2p{suffix}.pth"))
        torch.save(self.net_p2c.state_dict(), os.path.join(Config.OUTPUT_DIR, f"net_p2c{suffix}.pth"))

        # Coordinate grids at native resolutions
        grid_proj_hd = get_base_grid(Config.H_PROJ, Config.W_PROJ)
        grid_cam_hd = get_base_grid(Config.H_CAM, Config.W_CAM)

        # Network inference
        p_coords_hd = self.net_c2p(grid_cam_hd)
        c_coords_hd = self.net_p2c(grid_proj_hd)

        # Load full-res images
        if img_path is None:
            t_cam = load_image_tensor(Config.CAM_IMAGE_PATH, (Config.W_CAM, Config.H_CAM))
            t_proj = load_image_tensor(Config.REF_PATTERN_PATH, (Config.W_PROJ, Config.H_PROJ))
            img_without_distortion = self.generate_cam_without_distortion(Config.CAM_IMAGE_PATH, Config.REF_PATTERN_PATH)
        else:
            t_proj = load_image_tensor(img_path, (Config.W_PROJ, Config.H_PROJ))
            img_without_distortion = self.generate_cam_without_distortion(Config.CAM_IMAGE_PATH, Config.REF_PATTERN_PATH)

        if img_path is None:
            fake_proj = F.grid_sample(t_cam, c_coords_hd, align_corners=True)

        pre_warped = F.grid_sample(img_without_distortion, c_coords_hd, align_corners=True)
        fake_cam = F.grid_sample(t_proj, p_coords_hd, align_corners=True)

        if img_path is None:
            self.save_tensor(fake_proj, f"fake_proj{suffix}.png")
            self.save_tensor(pre_warped, f"pre_warped{suffix}.png")
            self.save_tensor(fake_cam, f"fake_cam{suffix}.png")
        else:
            self.save_tensor(pre_warped, f"pre_warped{suffix}.png")
            self.save_tensor(fake_cam, f"fake_cam{suffix}.png")

        print("All HD results and models saved.")

        # Quantitative metrics vs. ground truth
        pre_warped_path = os.path.join(Config.OUTPUT_DIR, f"pre_warped.png")
        pre_warped_ground_truth_path = "./data/3_image_prewarped.png"
        SSIM_pre_warped, PSNR_pre_warped, RMSE_pre_warped = self.calculate_metrics(pre_warped_path, pre_warped_ground_truth_path, apply_mask=True)
        print(f"Pre-warped  -- SSIM: {SSIM_pre_warped:.4f}, PSNR: {PSNR_pre_warped:.4f}, RMSE: {RMSE_pre_warped:.4f}")

        fake_cam_path = os.path.join(Config.OUTPUT_DIR, f"fake_cam.png")
        cam_ground_truth_path = Config.CAM_IMAGE_PATH
        SSIM_simulated, PSNR_simulated, RMSE_simulated = self.calculate_metrics(fake_cam_path, cam_ground_truth_path)
        print(f"Camera recon  -- SSIM: {SSIM_simulated:.4f}, PSNR: {PSNR_simulated:.4f}, RMSE: {RMSE_simulated:.4f}")

        fake_proj_path = os.path.join(Config.OUTPUT_DIR, f"fake_proj.png")
        proj_ground_truth_path = Config.REF_PATTERN_PATH
        SSIM_simulated_proj, PSNR_simulated_proj, RMSE_simulated_proj = self.calculate_metrics(fake_proj_path, proj_ground_truth_path)
        print(f"Projector recon -- SSIM: {SSIM_simulated_proj:.4f}, PSNR: {PSNR_simulated_proj:.4f}, RMSE: {RMSE_simulated_proj:.4f}")

    def calculate_metrics(self, img1_path, img2_path, apply_mask=False):
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        import numpy as np

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"[Warning] Could not read images for metric calculation: {img1_path} or {img2_path}")
            return 0, 0, 0

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        if apply_mask:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray2, 1, 255, cv2.THRESH_BINARY)
            mask_bool = mask > 0
            cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "metric_mask.png"), mask)
            print("[Debug] Saved metric mask: metric_mask.png")

            diff = img1.astype(np.float64) - img2.astype(np.float64)
            mse = np.mean((diff[mask_bool]) ** 2) if np.any(mask_bool) else 0
            rmse_val = np.sqrt(mse)

            if mse > 0:
                psnr_val = 10 * np.log10((255.0 ** 2) / mse)
            else:
                psnr_val = float('inf')

            _, ssim_img = ssim(img1, img2, channel_axis=-1, data_range=255, full=True)
            ssim_val = ssim_img[mask_bool].mean() if np.any(mask_bool) else 0.0

            return ssim_val, psnr_val, rmse_val

        # Global metrics
        ssim_val = ssim(img1, img2, channel_axis=-1, data_range=255)
        psnr_val = psnr(img1, img2, data_range=255)
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        rmse_val = np.sqrt(mse)

        return ssim_val, psnr_val, rmse_val

    def generate_cam_without_distortion(self, img_with_distortion_path, img_without_distortion_path):
        img_without_distortion = cv2.imread(img_without_distortion_path)
        img_without_distortion = cv2.cvtColor(img_without_distortion, cv2.COLOR_BGR2RGB)

        _, mask = create_mask(img_with_distortion_path)
        x, y, w_rect, h_rect = cv2.boundingRect(mask)

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

        mask_cam, _ = create_mask(Config.CAM_IMAGE_PATH, (Config.W_CAM, Config.H_CAM))

        grid_proj_hd = get_base_grid(Config.H_PROJ, Config.W_PROJ)

        # Map projector pixels to camera coordinates
        c_coords_hd = self.net_p2c(grid_proj_hd)

        # Sampling: a white projector image, warped through c_coords,
        # tells us which projector pixels land on valid camera content.
        mask_on_proj = F.grid_sample(mask_cam, c_coords_hd, align_corners=True)

        mask_np = (mask_on_proj.squeeze().cpu().numpy() * 255).astype(np.uint8)

        save_path = os.path.join(Config.OUTPUT_DIR, "Final_Mask_on_Projector_Space" + (f".iter{iteration}" if iteration is not None else "") + ".png")
        cv2.imwrite(save_path, mask_np)
        print(f"Projector mask saved to: {save_path}")

        # Overlay mask on projector image for visual check
        img_proj = cv2.imread(Config.REF_PATTERN_PATH)
        img_proj = cv2.resize(img_proj, (Config.W_PROJ, Config.H_PROJ))

        mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0
        mask_color[:, :, 1] = 0  # keep only red channel

        overlay = cv2.addWeighted(img_proj, 0.7, mask_color, 0.3, 0)
        cv2.imwrite(os.path.join(Config.OUTPUT_DIR, "Debug_Mask_Overlay_on_Proj" + (f".iter{iteration}" if iteration is not None else "") + ".png"), overlay)

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

    MODE = sys.argv[1] if len(sys.argv) > 1 else "inference"

    warper = NeuralWarper()

    if MODE == "train":
        warper.train()
    elif MODE == "inference":
        warper.run_inference(img_path=None)
    else:
        print("Unknown mode. Use 'train' or 'inference'.")