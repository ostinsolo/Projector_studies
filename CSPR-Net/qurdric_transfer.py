import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyvista as pv
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. Math & geometry core
# =============================================================================

def get_camera_matrices(fov_deg, res_w, res_h):
    """Compute intrinsic matrix K and its inverse from FOV and resolution."""
    f = 0.5 * res_w / np.tan(np.deg2rad(fov_deg) / 2)
    K = np.array([[f, 0, res_w / 2], [0, f, res_h / 2], [0, 0, 1]])
    return K, np.linalg.inv(K)

def get_camera_matrices_physical(f_mm, res_w, res_h, sensor_diag_inch="1/1.8"):
    """
    Compute K from physical focal length, resolution, and sensor size.

    f_mm: focal length in mm (e.g. 10.0)
    res_w, res_h: image resolution in pixels (e.g. 3072, 1728)
    sensor_diag_inch: optical format string like "1/1.8", "1/2", "2/3", etc.,
                      or a numeric diagonal length in mm.
    """

    # Sensor diagonal lookup (mm)
    sensor_map = {
        "1/1.8": 8.933,
        "1/2": 8.000,
        "1/2.3": 7.700,
        "1/2.5": 7.182,
        "1/2.7": 6.604,
        "1/3": 6.000,
        "2/3": 11.000,
        "1": 16.000
    }

    if isinstance(sensor_diag_inch, str) and sensor_diag_inch in sensor_map:
        diag_mm = sensor_map[sensor_diag_inch]
    elif isinstance(sensor_diag_inch, (int, float)):
        diag_mm = float(sensor_diag_inch)
    else:
        print(f"Warning: unknown sensor format {sensor_diag_inch}, falling back to 1/1.8\" (8.93mm)")
        diag_mm = 8.933

    # Sensor physical width from aspect ratio: W = D / sqrt(1 + r^2)
    ratio = res_h / res_w
    sensor_w_mm = diag_mm / np.sqrt(1 + ratio**2)

    # Pixel-size focal length
    f_pix = f_mm * (res_w / sensor_w_mm)

    # Principal point at image center
    c_x = res_w / 2.0
    c_y = res_h / 2.0

    K = np.array([
        [f_pix, 0,     c_x],
        [0,     f_pix, c_y],
        [0,     0,     1  ]
    ])

    pixel_size_um = (sensor_w_mm / res_w) * 1000
    print(f"--- Sensor / intrinsics ---")
    print(f"Sensor width: {sensor_w_mm:.3f} mm")
    print(f"Pixel size:   {pixel_size_um:.2f} um")
    print(f"Focal length (px): fx={f_pix:.2f}, fy={f_pix:.2f}")
    fov_x = 2 * np.degrees(np.arctan((res_w / 2) / f_pix))
    fov_y = 2 * np.degrees(np.arctan((res_h / 2) / f_pix))
    print(f"Estimated FOV: x={fov_x:.2f} deg, y={fov_y:.2f} deg")
    return K, np.linalg.inv(K)

def get_projector_matrices(throw_ratio, res_w, res_h, offset_percent=0):
    """
    Compute projector intrinsics K from throw ratio, resolution, and offset.

    throw_ratio: e.g. 1.2
    offset_percent: vertical lens shift. 0 = centered, 100 = bottom-aligned (desktop).
    """

    f_x = throw_ratio * res_w
    f_y = f_x  # square pixels

    c_x = res_w / 2.0

    # Offset shifts the principal point vertically.
    # cy = h/2 * (1 + offset/100) — so 100% offset pushes cy to the bottom edge.
    shift_factor = offset_percent / 100.0
    c_y = (res_h / 2.0) * (1.0 + shift_factor)

    K = np.array([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1  ]
    ])

    fov_x = 2 * np.degrees(np.arctan((res_w / 2) / f_x))
    fov_y = 2 * np.degrees(np.arctan((res_h / 2) / f_y))
    print(f"Projector FOV: x={fov_x:.2f} deg, y={fov_y:.2f} deg")
    return K, np.linalg.inv(K)

def build_pose(C, R):
    """Build extrinsic matrix [R|t] from camera center C and rotation R."""
    t = -R @ C
    return np.hstack([R, t[:, np.newaxis]])

def get_camera_rays(K_inv, R_T, C, w, h):
    """Generate a ray for every pixel in the image."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)
    cam_coords = (K_inv @ uv1.reshape(-1, 3).T).T.reshape(h, w, 3)
    directions = (R_T @ cam_coords.reshape(-1, 3).T).T.reshape(h, w, 3)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.tile(C, (h, w, 1))
    return origins, directions

def get_camera_rays_batch(K_inv, R_T, C, u_coords, v_coords):
    """Generate rays for a batch of pixel coordinates."""
    n = u_coords.size
    uv1 = np.stack([u_coords, v_coords, np.ones(n)], axis=-1)
    cam_coords = (K_inv @ uv1.T).T
    directions = (R_T @ cam_coords.T).T
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.tile(C, (n, 1))
    return origins, directions

def intersect_ray_quadric(ray_O, ray_D, coeffs):
    """Ray–quadric intersection. Returns t values (inf where no hit)."""
    A, B, C, D, E, F, G, H, I, J = coeffs
    ox, oy, oz = ray_O[..., 0], ray_O[..., 1], ray_O[..., 2]
    dx, dy, dz = ray_D[..., 0], ray_D[..., 1], ray_D[..., 2]

    term_a = (A * dx**2 + B * dy**2 + C * dz**2 + D * dx * dy + E * dx * dz + F * dy * dz)
    term_b = (2*A*ox*dx + 2*B*oy*dy + 2*C*oz*dz + D*(ox*dy + oy*dx) + E*(ox*dz + oz*dx) + F*(oy*dz + oz*dy) + G*dx + H*dy + I*dz)
    term_c = (A*ox**2 + B*oy**2 + C*oz**2 + D*ox*oy + E*ox*oz + F*oy*oz + G*ox + H*oy + I*oz + J)

    term_a[np.abs(term_a) < 1e-9] = 1e-9

    delta = term_b**2 - 4 * term_a * term_c

    t = np.full_like(delta, np.inf)
    mask = delta >= 0

    sqrt_delta = np.sqrt(delta[mask])
    t1 = (-term_b[mask] - sqrt_delta) / (2 * term_a[mask])
    t2 = (-term_b[mask] + sqrt_delta) / (2 * term_a[mask])

    t1[t1 < 1e-3] = np.inf
    t2[t2 < 1e-3] = np.inf
    t[mask] = np.minimum(t1, t2)
    return t

def project_points(K, Rt, X_world):
    """Project 3D world points to 2D image plane."""
    if X_world.shape[0] == 0: return np.array([])
    N = X_world.shape[0]
    X_hom = np.hstack([X_world, np.ones((N, 1))])
    p_hom = (K @ Rt @ X_hom.T).T
    p_img = p_hom[:, :2] / (p_hom[:, 2, np.newaxis] + 1e-8)
    return p_img

def gen_checkerboard(w, h, n_squares=10):
    """Generate a checkerboard pattern image."""
    img = np.zeros((h, w), dtype=np.uint8)
    square_size = w / n_squares
    for i in range(h):
        for j in range(w):
            if (int(i / square_size) + int(j / square_size)) % 2 == 0:
                img[i, j] = 255
    return img

def create_mask(img, size=None):
    """Extract foreground mask from an image via threshold + contour fill."""
    if img is None:
        raise ValueError(f"Input image is None")
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def get_quadric_mesh_general(coeffs, bounds=(-2, 2, -2, 2, -2, 2), dims=(100, 100, 100)):
    """Extract an isosurface mesh for the implicit quadric f(x,y,z)=0."""
    grid = pv.ImageData(
        dimensions=dims,
        origin=(bounds[0], bounds[2], bounds[4]),
        spacing=(
            (bounds[1]-bounds[0])/(dims[0]-1),
            (bounds[3]-bounds[2])/(dims[1]-1),
            (bounds[5]-bounds[4])/(dims[2]-1),
        )
    )
    x, y, z = grid.points.T
    A, B, C, D, E, F, G, H, I, J = coeffs
    values = (A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J)
    grid.point_data["values"] = values
    try:
        mesh = grid.contour([0])
        return mesh
    except Exception as e:
        print(f"Failed to generate quadric mesh: {e}")
        return None

# =============================================================================
# 2. PyVista 3D visualization
# =============================================================================

def visualize_with_pyvista(
        CAM_C, CAM_R_T, K_cam, CAM_W, CAM_H,
        PROJ_C, PROJ_R_T, K_proj, PROJ_W, PROJ_H,
        QUADRIC_COEFFS,
        texture_proj=None, texture_cam=None,
        save_path=None,
        high_res_png_path=None):

    plotter = pv.Plotter(window_size=[2560, 1440])
    plotter.set_background('white')
    plotter.enable_anti_aliasing('ssaa')

    print(f"\n[PyVista] Starting hi-res 3D visualization...")
    if save_path: print(f"  Vector/PDF target: {save_path}")
    if high_res_png_path: print(f"  Hi-res PNG target: {high_res_png_path}")

    # --- Quadric surface ---
    mesh = get_quadric_mesh_general(QUADRIC_COEFFS, bounds=(-5, 5, -5, 5, -3, 3))
    if mesh is not None:
        surf_props = dict(color='cyan', opacity=1.0, specular=0.5, show_edges=False)
        plotter.add_mesh(mesh, **surf_props)
        wire_props = dict(color='blue', opacity=1.0, style='wireframe')
        plotter.add_mesh(mesh, **wire_props)

    # --- Camera & projector frustums ---
    def add_device_viz(name, C, R_T, K_inv, w, h, scale, color, img_texture=None):
        corners_pix = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]], dtype=float)
        corners_cam = (K_inv @ corners_pix.T).T
        corners_dir = (R_T @ corners_cam.T).T
        corners_world = C + corners_dir * scale

        plotter.add_mesh(pv.Sphere(radius=0.05, center=C), color=color)

        for i in range(4):
            line = pv.Line(C, corners_world[i])
            plotter.add_mesh(line, color=color, line_width=4)

        faces = np.hstack([[4, 0, 1, 2, 3]])
        plane = pv.PolyData(corners_world, faces)

        if img_texture is not None:
            if len(img_texture.shape) == 2:
                img_texture = cv2.cvtColor(img_texture, cv2.COLOR_GRAY2RGB)
            img_texture = np.flipud(img_texture)
            tex = pv.numpy_to_texture(img_texture)
            tex.InterpolateOn()
            plane.texture_map_to_plane(inplace=True, origin=corners_world[0], point_u=corners_world[1], point_v=corners_world[3])
            plotter.add_mesh(plane, texture=tex, opacity=0.9, show_edges=True, edge_color=color, line_width=3)
        else:
            plotter.add_mesh(plane, color=color, opacity=0.2, show_edges=True, line_width=3)
        return corners_world

    K_cam_inv = np.linalg.inv(K_cam)
    K_proj_inv = np.linalg.inv(K_proj)
    frustum_scale = 0.8

    add_device_viz("Projector", PROJ_C, PROJ_R_T, K_proj_inv, PROJ_W, PROJ_H, frustum_scale, 'red', texture_proj)
    add_device_viz("Camera", CAM_C, CAM_R_T, K_cam_inv, CAM_W, CAM_H, frustum_scale, 'green', texture_cam)

    # --- Ray bundles (sampled grid) ---
    grid_nx, grid_ny = 6, 8
    gx = np.linspace(0, PROJ_W-1, grid_nx)
    gy = np.linspace(0, PROJ_H-1, grid_ny)
    U, V = np.meshgrid(gx, gy)
    u_flat, v_flat = U.flatten(), V.flatten()

    p_origins, p_dirs = get_camera_rays_batch(K_proj_inv, PROJ_R_T, PROJ_C, u_flat, v_flat)
    t_vals = intersect_ray_quadric(p_origins, p_dirs, QUADRIC_COEFFS)

    valid = t_vals != np.inf
    points_surface = p_origins[valid] + t_vals[valid, np.newaxis] * p_dirs[valid]

    # Rays: projector -> surface
    lines_p2s = []
    for i in range(len(points_surface)):
        lines_p2s.append(PROJ_C)
        lines_p2s.append(points_surface[i])
    if len(lines_p2s) > 0:
        lines_flat = np.hstack([[2, 2*i, 2*i+1] for i in range(len(points_surface))])
        pd_rays = pv.PolyData(np.array(lines_p2s))
        pd_rays.lines = lines_flat
        plotter.add_mesh(pd_rays, color='red', opacity=0.6, line_width=4)

    # Rays: surface -> camera
    lines_s2c = []
    for i in range(len(points_surface)):
        lines_s2c.append(points_surface[i])
        lines_s2c.append(CAM_C)
    if len(lines_s2c) > 0:
        lines_flat_2 = np.hstack([[2, 2*i, 2*i+1] for i in range(len(points_surface))])
        pd_rays_2 = pv.PolyData(np.array(lines_s2c))
        pd_rays_2.lines = lines_flat_2
        plotter.add_mesh(pd_rays_2, color='green', opacity=0.5, line_width=4)

    if len(points_surface) > 0:
         plotter.add_mesh(pv.PolyData(points_surface), color='black', point_size=12, render_points_as_spheres=True)

    # --- Camera setup ---
    plotter.add_axes()
    focal_point = [0, -1.2, 0]
    position = [-5, -10, 8]
    view_up = [0, 0, 1]
    plotter.camera_position = [position, focal_point, view_up]
    plotter.enable_parallel_projection()
    plotter.render()

    # --- Export ---
    if save_path:
        success = plotter.save_graphic(save_path)
        if success:
            print(f"[PyVista] Vector graphic saved to: {save_path}")
        else:
            print(f"[PyVista] Save failed — check file extension.")

    if high_res_png_path:
        plotter.screenshot(high_res_png_path, scale=3, return_img=False)
        print(f"[PyVista] Hi-res PNG saved to: {high_res_png_path}")

    print("[PyVista] Render done.")
    plotter.show()


# =============================================================================
# 3. Main
# =============================================================================

if __name__ == "__main__":
    CAM_W, CAM_H = 3072, 1728
    PROJ_W, PROJ_H = 1920, 1080

    # Vertical cylinder: x^2 + z^2 = R^2
    R_cylinder = 2.0
    # Coefficients: [A, B, C, D, E, F, G, H, I, J]
    QUADRIC_COEFFS = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -(R_cylinder**2)]

    # Alt: cone — x^2 + y^2 - (k*z)^2 = 0
    # k_cone = np.tan(np.deg2rad(30))
    # QUADRIC_COEFFS = [1.0, 1.0, -(k_cone**2), 0.0, 0.0, 0.0, 0.0, 0.0, k_cone, -k_cone]

    TARGET = np.array([0.0, -1.2, 0.0])
    WORLD_UP = np.array([0.0, 0.0, 1.0])

    # Camera
    f_mm = 10.0
    sensor_format = "1/1.8"
    CAM_C = np.array([0.0, -7, 1])
    cam_fwd = (TARGET - CAM_C)
    cam_fwd /= np.linalg.norm(cam_fwd)

    cam_right = np.cross(cam_fwd, WORLD_UP)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_fwd)
    CAM_R = np.stack([cam_right, cam_up, cam_fwd], axis=0)
    K_cam, K_cam_inv = get_camera_matrices_physical(f_mm, CAM_W, CAM_H, sensor_format)
    Rt_cam = build_pose(CAM_C, CAM_R)
    CAM_R_T = CAM_R.T

    # Projector
    PROJ_FOV = 1
    PROJ_C = np.array([0.0, -4.5, 2])
    proj_fwd = (TARGET - PROJ_C); proj_fwd /= np.linalg.norm(proj_fwd)
    proj_right = np.cross(proj_fwd, WORLD_UP); proj_right /= np.linalg.norm(proj_right)
    proj_up = np.cross(proj_right, proj_fwd)
    PROJ_R = np.stack([proj_right, proj_up, proj_fwd], axis=0)

    tr = 1.2
    w = 1920
    h = 1080
    K_proj, K_proj_inv = get_projector_matrices(tr, w, h, offset_percent=0)

    Rt_proj = build_pose(PROJ_C, PROJ_R)
    PROJ_R_T = PROJ_R.T

    print("Computing ray-traced mapping (camera -> surface -> projector)...")
    cam_rays_O, cam_rays_D = get_camera_rays(K_cam_inv, CAM_R_T, CAM_C, CAM_W, CAM_H)
    t_cam = intersect_ray_quadric(cam_rays_O, cam_rays_D, QUADRIC_COEFFS)
    valid_mask_cam = t_cam != np.inf
    X_world_cam = cam_rays_O[valid_mask_cam] + t_cam[valid_mask_cam, np.newaxis] * cam_rays_D[valid_mask_cam]
    uv_proj = project_points(K_proj, Rt_proj, X_world_cam)
    cam_to_proj_map_u = np.full((CAM_H, CAM_W), -1.0, dtype=np.float32)
    cam_to_proj_map_v = np.full((CAM_H, CAM_W), -1.0, dtype=np.float32)
    if uv_proj.shape[0] > 0:
        cam_to_proj_map_u[valid_mask_cam] = uv_proj[:, 0]
        cam_to_proj_map_v[valid_mask_cam] = uv_proj[:, 1]

    print("Computing ray-traced mapping (projector -> surface -> camera)...")
    proj_rays_O, proj_rays_D = get_camera_rays(K_proj_inv, PROJ_R_T, PROJ_C, PROJ_W, PROJ_H)
    t_proj = intersect_ray_quadric(proj_rays_O, proj_rays_D, QUADRIC_COEFFS)
    valid_mask_proj = t_proj != np.inf
    X_world_proj = proj_rays_O[valid_mask_proj] + t_proj[valid_mask_proj, np.newaxis] * proj_rays_D[valid_mask_proj]
    uv_cam = project_points(K_cam, Rt_cam, X_world_proj)
    proj_to_cam_map_u = np.full((PROJ_H, PROJ_W), -1.0, dtype=np.float32)
    proj_to_cam_map_v = np.full((PROJ_H, PROJ_W), -1.0, dtype=np.float32)
    if uv_cam.shape[0] > 0:
        proj_to_cam_map_u[valid_mask_proj] = uv_cam[:, 0]
        proj_to_cam_map_v[valid_mask_proj] = uv_cam[:, 1]

    print("Mapping done. Processing images...")
    for idx in [1, 2, 3]:
        print(f"\n[{idx}/3] Processing...")
        pro = cv2.imread(f"./sim_data/{idx}_pro.png")
        if pro is None:
            print(f"Warning: could not read ./sim_data/{idx}_pro.png")
            continue
        cam = pro.copy()

        image_distorted = cv2.remap(pro, cam_to_proj_map_u, cam_to_proj_map_v, cv2.INTER_LINEAR, borderValue=0)

        # Camera-view mask + crop/scale to fill the valid region
        camera_mask = create_mask(image_distorted, size=(CAM_W, CAM_H))
        x, y, w_rect, h_rect = cv2.boundingRect(camera_mask)

        if w_rect > 0 and h_rect > 0:
            cam_resized = cv2.resize(cam, (w_rect, h_rect), interpolation=cv2.INTER_LANCZOS4)
            new_cam = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            new_cam[y:y+h_rect, x:x+w_rect] = cam_resized
            cam = new_cam
        else:
            print("Warning: empty mask, skipping crop/scale.")

        image_prewarped = cv2.remap(cam, proj_to_cam_map_u, proj_to_cam_map_v, cv2.INTER_LINEAR, borderValue=0)
        image_verified = cv2.remap(image_prewarped, cam_to_proj_map_u, cam_to_proj_map_v, cv2.INTER_LINEAR, borderValue=0)

        cv2.imwrite(f"./data/{idx}_image_distorted.png", image_distorted)
        cv2.imwrite(f"./data/{idx}_image_prewarped.png", image_prewarped)
        cv2.imwrite(f"./data/{idx}_image_verified.png", image_verified)
        cv2.imwrite(f"./sim_data/{idx}_image_distorted.png", image_distorted)

        if idx == 3:
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 2, 1)
            plt.title("Original (naive projector input)")
            plt.imshow(cv2.cvtColor(pro, cv2.COLOR_BGR2RGB))

            plt.subplot(2, 2, 2)
            plt.title("Distorted (camera view)")
            plt.imshow(cv2.cvtColor(image_distorted, cv2.COLOR_BGR2RGB))

            plt.subplot(2, 2, 3)
            plt.title("Pre-warped projector input")
            plt.imshow(cv2.cvtColor(image_prewarped, cv2.COLOR_BGR2RGB))

            plt.subplot(2, 2, 4)
            plt.title("Corrected (verified camera view)")
            plt.imshow(cv2.cvtColor(image_verified, cv2.COLOR_BGR2RGB))
            plt.tight_layout()
            plt.savefig(f"./sim_data/{idx}_result_plot.png")

            texture_img = cv2.imread(f"./data/3_image_distorted.png")
            if texture_img is not None:
                visualize_with_pyvista(
                    CAM_C, CAM_R_T, K_cam, CAM_W, CAM_H,
                    PROJ_C, PROJ_R_T, K_proj, PROJ_W, PROJ_H,
                    QUADRIC_COEFFS,
                    texture_proj=cv2.imread(f"./sim_data/3_pro.png"),
                    texture_cam=texture_img,
                    save_path=None,
                    high_res_png_path=None
                )