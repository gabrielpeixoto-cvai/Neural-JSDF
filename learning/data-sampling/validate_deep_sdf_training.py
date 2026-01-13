import numpy as np
from pathlib import Path
import os
import trimesh
import mujoco
from typing import Tuple, List, Dict
import torch
import random

# --- ASSUMPTION: Decoder is importable from your training setup ---
try:
    from deep_sdf_decoder import Decoder
except ImportError:
    print("FATAL: Could not import Decoder. Ensure deep_sdf_decoder.py is accessible.")
    sys.exit(1)

# --- 1. Utility Functions (Consolidated from Data Generation) ---


# Mock Model & Data loading (replace with your actual paths/imports)
def load_data(robot_file_path_str: str) -> Tuple[mujoco.MjModel, List[str]]:
    """Loads the Mujoco model and returns the model object and a list of body names."""
    try:
        robot_model = mujoco.MjModel.from_xml_path(str(robot_file_path_str))
    except Exception as e:
        print(f"Error loading robot file: {e}")
        raise

    mesh_names = []
    # Loop over all bodies in the model to get body names with geometry
    for body_id in range(1, robot_model.nbody):  # Start from 1 to skip world body
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if get_body_mesh(robot_model, body_id) is not None:
            mesh_names.append(body_name)

    return robot_model, mesh_names


def get_geom_mesh_data_mjcf(model, geom_id):
    """Extracts mesh vertices and faces for a given geom ID from a Mujoco model."""
    geom_type = model.geom_type[geom_id]
    geom_dataid = model.geom_dataid[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = geom_dataid
        vert_start = model.mesh_vertadr[mesh_id]
        vert_end = vert_start + model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_end = face_start + model.mesh_facenum[mesh_id]
        return (
            model.mesh_vert[vert_start:vert_end],
            model.mesh_face[face_start:face_end],
        )
    else:
        return None, None


def get_body_mesh(model, body_id):
    """Collects and merges all mesh geoms belonging to a body into a single Trimesh."""
    meshes = []
    if model.body_geomadr[body_id] < 0:
        return None
    geom_id_start = model.body_geomadr[body_id]

    for i in range(model.body_geomnum[body_id]):
        gid = geom_id_start + i
        vertices, faces = get_geom_mesh_data_mjcf(model, gid)
        if vertices is None:
            continue

        pos = model.geom_pos[gid]
        quat = model.geom_quat[gid]
        transform = trimesh.transformations.quaternion_matrix(quat)
        transform[:3, 3] = pos
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.apply_transform(transform)
        meshes.append(mesh)

    return trimesh.util.concatenate(meshes) if meshes else None


def get_joint_limits(m: mujoco.MjModel):
    """Extracts the joint limits (min and max position)."""
    joint_limit_min, joint_limit_max = [], []
    for joint_id in range(m.njnt):
        if m.jnt_limited[joint_id] == 1:
            qpos_range = m.jnt_range[joint_id]
            joint_limit_min.append(qpos_range[0])
            joint_limit_max.append(qpos_range[1])
    return np.array(joint_limit_min), np.array(joint_limit_max)


def mujoco_mesh_fk(robot_model, qpos) -> List[trimesh.Trimesh]:
    """Computes FK and returns all world-transformed meshes for visualization."""
    d = mujoco.MjData(robot_model)
    d.qpos[:] = qpos
    mujoco.mj_forward(robot_model, d)

    mesh_data = []
    for body_id in range(1, robot_model.nbody):
        mesh = get_body_mesh(robot_model, body_id)
        if mesh is None:
            continue

        body_pos = d.xpos[body_id]
        body_mat = d.xmat[body_id].reshape(3, 3)

        T = np.eye(4)
        T[:3, :3] = body_mat
        T[:3, 3] = body_pos

        mesh_world = mesh.copy()
        mesh_world.apply_transform(T)
        mesh_data.append(mesh_world)
    return mesh_data


def map_scalar_to_color_custom(data, zero_tolerance=1e-4):
    """Maps scalar data to Red (Negative/Inside), Black (Zero/Surface), and Blue (Positive/Outside) RGBA colors."""
    N = len(data)
    colors = np.zeros((N, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Fully opaque

    idx_negative = data < -zero_tolerance
    colors[idx_negative, 0] = 255  # R (Red for Inside)

    idx_positive = data > zero_tolerance
    colors[idx_positive, 2] = 255  # B (Blue for Outside)

    # Black remains for surface
    return colors


def display_mujoco_meshes_and_sdfs(
    transformed_meshes: List[trimesh.Trimesh],
    points_world: np.ndarray,
    sdfs: np.ndarray,
    point_size: float = 5.0,
):
    """Displays meshes and sampled SDF points."""
    if not transformed_meshes:
        print("No meshes to display.")
        return

    point_colors = map_scalar_to_color_custom(sdfs)
    sdf_point_cloud = trimesh.points.PointCloud(
        points_world, colors=point_colors, metadata={"point_size": point_size}
    )

    scene = trimesh.Scene()
    scene.add_geometry(sdf_point_cloud)

    for mesh in transformed_meshes:
        if mesh.visual.kind == "face":
            mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh)
            mesh.visual.face_colors = np.array(
                [150, 150, 150, 150], dtype=np.uint8
            )  # Gray and transparent
        else:
            mesh.visual.face_colors = np.array(
                [150, 150, 150, 150], dtype=np.uint8
            )  # Gray and transparent

        scene.add_geometry(mesh)

    print(
        "\nDisplaying results. SDF Color Key: Red=Inside, Black=Surface, Blue=Outside"
    )
    try:
        scene.show(smooth=False)
    except Exception as e:
        print(f"\n[ERROR] Trimesh viewer failed: {e}")


# --- 2. Model and Data Loading ---


def load_trained_model(
    experiment_dir: str,
    link_name: str,
    latent_size: int,
    snapshot_filename: str = "latest.pth",
) -> Tuple[Decoder, torch.Tensor]:
    """
    Loads the trained Decoder and the single latent vector.

    Returns: (decoder_model, single_latent_vector)
    """
    exp_path = Path(experiment_dir)

    # 1. Load Decoder State Dict
    model_params_path = exp_path / "ModelParams" / snapshot_filename
    if not model_params_path.exists():
        raise FileNotFoundError(f"Model parameters not found at {model_params_path}")

    checkpoint = torch.load(model_params_path)
    decoder = Decoder(
        latent_size,
        **{
            "dims": [512, 512, 512, 512, 512, 512, 512, 512],
            "latent_in": [4],
            "weight_norm": True,
            "latent_dropout": True,
            "xyz_in_all": False,
        },
    ).cuda()
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    print(f"Loaded Decoder model from epoch {checkpoint['epoch']}.")

    # 2. Load Latent Code
    latent_codes_path = exp_path / "LatentCodes" / snapshot_filename
    if not latent_codes_path.exists():
        raise FileNotFoundError(f"Latent codes not found at {latent_codes_path}")

    latent_checkpoint = torch.load(latent_codes_path)
    # The saved state dict contains one embedding row (index 0)
    # We create a dummy Embedding layer to load the weights
    lat_vecs_dummy = torch.nn.Embedding(1, latent_size)
    lat_vecs_dummy.load_state_dict(latent_checkpoint["latent_codes"])

    # Extract the single latent vector (1, LATENT_SIZE) and move to GPU
    single_latent_vec = lat_vecs_dummy.weight.data.cuda().squeeze()
    print(f"Loaded single latent vector for link '{link_name}'.")

    return decoder, single_latent_vec


# --- 3. Transformation Function ---


def get_world_to_local_transform(
    robot_model: mujoco.MjModel, d: mujoco.MjData, link_name: str
) -> np.ndarray:
    """
    Computes the 4x4 homogeneous transformation matrix T_World_to_Local (T_W_L^-1).
    P_Local = T_W_L^-1 @ P_World
    """
    mujoco_body_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

    if mujoco_body_id < 0:
        raise ValueError(f"Link name '{link_name}' not found in Mujoco model.")

    # Transformation from Local (L) to World (W): T_W_L
    R_W_L = d.xmat[mujoco_body_id].reshape(3, 3)  # Rotation (3x3)
    P_W_L = d.xpos[mujoco_body_id]  # Position (3,)

    # The inverse T_L_W = T_W_L^-1 is:
    # [ R_W_L^T  | -R_W_L^T * P_W_L ]
    # [   0      |        1         ]

    R_L_W = R_W_L.T
    P_L_W = -R_L_W @ P_W_L

    T_W_L_inv = np.eye(4)
    T_W_L_inv[:3, :3] = R_L_W
    T_W_L_inv[:3, 3] = P_L_W

    return T_W_L_inv


# --- 4. Main Validation Script ---
def get_link_world_center(robot_model, d, link_name: str) -> np.ndarray:
    """Returns the (3,) world position of the specified link."""
    body_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if body_id == -1:
        raise ValueError(f"Link '{link_name}' not found in model.")
    # Return a copy to prevent accidental modification of MjData
    return d.xpos[body_id].copy()


def get_bbox(vertices: np.ndarray) -> Dict[str, float]:
    """Calculates the bounding box for a set of vertices."""
    return {
        "xmin": np.min(vertices[:, 0]),
        "xmax": np.max(vertices[:, 0]),
        "ymin": np.min(vertices[:, 1]),
        "ymax": np.max(vertices[:, 1]),
        "zmin": np.min(vertices[:, 2]),
        "zmax": np.max(vertices[:, 2]),
    }


def scale_bbox(bbox: Dict[str, float], delta: float) -> Dict[str, float]:
    """Expands or shrinks a bounding box by a fixed delta."""
    return {
        "xmin": bbox["xmin"] - delta,
        "xmax": bbox["xmax"] + delta,
        "ymin": bbox["ymin"] - delta,
        "ymax": bbox["ymax"] + delta,
        "zmin": bbox["zmin"] - delta,
        "zmax": bbox["zmax"] + delta,
    }


def sample_bbox(bbox: Dict[str, float], n_points: int) -> np.ndarray:
    """Uniformly samples points within a bounding box."""
    return np.random.uniform(
        low=[bbox["xmin"], bbox["ymin"], bbox["zmin"]],
        high=[bbox["xmax"], bbox["ymax"], bbox["zmax"]],
        size=(n_points, 3),
    )


"""
def validate_model(
    robot_file: str,
    experiment_dir: str,
    target_link_name: str,
    latent_size: int,
    n_sample_points: int = 100000,
):

    # 1. Load Robot Model
    robot_model, mesh_names = load_data(robot_file)
    q_min, q_max = get_joint_limits(robot_model)
    d = mujoco.MjData(robot_model)

    # 2. Load Network
    try:
        decoder, latent_vec = load_trained_model(
            experiment_dir, target_link_name, latent_size
        )
    except FileNotFoundError as e:
        print(f"Validation failed: {e}")
        return

    # 3. Move Robot to a Random Configuration (FK)
    q_span = q_max - q_min
    q_rand = q_min + np.random.rand(robot_model.nq) * q_span

    # Ensure qpos is set and FK is run for correct body positions
    d.qpos[:] = q_rand
    mujoco.mj_forward(robot_model, d)

    print(f"\n--- Validation with Random Configuration ---")
    print(f"Joint Angles: {q_rand}")

    # Get all transformed meshes for visualization
    transformed_meshes = mujoco_mesh_fk(robot_model, q_rand)

    link_center = get_link_world_center(robot_model, d, target_link_name)

    # 4. Sample Data in World Coordinates
    # Bounding box: [-1.5, 1.5]m in all directions
    # world_bbox = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]])

    # Define a sampling extent (e.g., 20cm around the center)
    extent = 0.15

    # Create the world_bbox relative to the link center
    # world_bbox shape: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    world_bbox = np.array(
        [
            [link_center[0] - extent, link_center[0] + extent],
            [link_center[1] - extent, link_center[1] + extent],
            [link_center[2] - extent, link_center[2] + extent],
        ]
    )

    # Sample points uniformly in the world bounding box
    points_world = np.random.uniform(
        low=world_bbox[:, 0], high=world_bbox[:, 1], size=(n_sample_points, 3)
    ).astype(np.float32)

    print(f"Sampled {n_sample_points} points in World Frame...")

    # 5. Translate Points in World Coordinates to Local Coordinates
    try:
        T_W_L_inv = get_world_to_local_transform(robot_model, d, target_link_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Pad points_world with 1s for homogeneous coordinates (N, 4)
    points_world_h = np.hstack(
        [points_world, np.ones((n_sample_points, 1))]
    ).T  # (4, N)

    # P_Local_h = T_W_L_inv @ P_World_h
    points_local_h = T_W_L_inv @ points_world_h
    points_local = points_local_h[:3, :].T  # (N, 3)

    # 6. Execute Inference
    with torch.no_grad():
        # Convert local points to tensor and move to GPU
        points_local_tensor = torch.from_numpy(points_local).cuda().float()

        # Replicate latent vector to match batch size (N, L)
        # latent_vec is (1, L)
        batch_latent_vecs = latent_vec.expand(n_sample_points, -1)

        # Concatenate latent vector and points (N, L+3)
        input_data = torch.cat([batch_latent_vecs, points_local_tensor], dim=1)

        # Inference
        predicted_sdfs_tensor = decoder(input_data)

        # Move results back to CPU and convert to numpy
        predicted_sdfs = predicted_sdfs_tensor.cpu().numpy().squeeze()
        print(points_world)
        print(points_local)
        print(predicted_sdfs)

    # 7. Display the Result
    # We display the points in the World Frame but colored by the predicted SDF
    display_mujoco_meshes_and_sdfs(transformed_meshes, points_world, predicted_sdfs)
"""


def validate_model(
    robot_file: str,
    experiment_dir: str,
    target_link_name: str,
    latent_size: int,
    n_sample_points: int = 100000,
):
    # 1. Load Robot Model and Data
    robot_model, _ = load_data(robot_file)
    d = mujoco.MjData(robot_model)

    # 2. Randomize Configuration and Run Forward Kinematics
    q_min, q_max = get_joint_limits(robot_model)
    q_rand = q_min + np.random.rand(robot_model.nq) * (q_max - q_min)
    d.qpos[:] = q_rand
    mujoco.mj_forward(robot_model, d)

    # 3. Get World Transform for the target link (T_W_L)
    body_id = mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, target_link_name)
    body_pos = d.xpos[body_id]
    body_mat = d.xmat[body_id].reshape(3, 3)
    T_W_L = np.eye(4)
    T_W_L[:3, :3] = body_mat
    T_W_L[:3, 3] = body_pos

    # 4. Determine World Bounding Box
    local_mesh = get_body_mesh(robot_model, body_id)
    local_bbox = get_bbox(
        local_mesh.vertices
    )  # Using function from generate_local_dataset_mujoco.py

    # Create the 8 corners of the local bounding box
    corners_l = np.array(
        [
            [local_bbox["xmin"], local_bbox["ymin"], local_bbox["zmin"]],
            [local_bbox["xmin"], local_bbox["ymin"], local_bbox["zmax"]],
            [local_bbox["xmin"], local_bbox["ymax"], local_bbox["zmin"]],
            [local_bbox["xmin"], local_bbox["ymax"], local_bbox["zmax"]],
            [local_bbox["xmax"], local_bbox["ymin"], local_bbox["zmin"]],
            [local_bbox["xmax"], local_bbox["ymin"], local_bbox["zmax"]],
            [local_bbox["xmax"], local_bbox["ymax"], local_bbox["zmin"]],
            [local_bbox["xmax"], local_bbox["ymax"], local_bbox["zmax"]],
        ]
    )

    # Transform corners to World Frame: P_W = T_W_L @ P_L
    corners_l_h = np.hstack([corners_l, np.ones((8, 1))]).T
    corners_w = (T_W_L @ corners_l_h)[:3, :].T

    # Define a world-aligned bbox that contains the transformed link
    world_bbox = {
        "xmin": np.min(corners_w[:, 0]) - 0.05,
        "xmax": np.max(corners_w[:, 0]) + 0.05,
        "ymin": np.min(corners_w[:, 1]) - 0.05,
        "ymax": np.max(corners_w[:, 1]) + 0.05,
        "zmin": np.min(corners_w[:, 2]) - 0.05,
        "zmax": np.max(corners_w[:, 2]) + 0.05,
    }

    # 5. Sample Points in World Coordinates
    points_world = sample_bbox(world_bbox, n_sample_points).astype(np.float32)

    # 6. Transform World Points to Local Frame for Inference
    T_L_W = np.linalg.inv(T_W_L)
    points_world_h = np.hstack([points_world, np.ones((n_sample_points, 1))]).T
    points_local = (T_L_W @ points_world_h)[:3, :].T

    # 7. Execute Inference using local coordinates
    decoder, latent_vec = load_trained_model(
        experiment_dir, target_link_name, latent_size
    )
    with torch.no_grad():
        points_local_tensor = torch.from_numpy(points_local).cuda().float()
        batch_latent = latent_vec.expand(n_sample_points, -1)
        input_data = torch.cat([batch_latent, points_local_tensor], dim=1)
        predicted_sdfs = decoder(input_data).cpu().numpy().squeeze()

    # 8. Display result (World points colored by predicted SDFs)
    transformed_meshes = mujoco_mesh_fk(robot_model, q_rand)
    display_mujoco_meshes_and_sdfs(transformed_meshes, points_world, predicted_sdfs)


if __name__ == "__main__":

    # --- CONFIGURATION ---

    # 📌 1. Update this to your robot model file
    ROBOT_FILE_PATH = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml"

    # 📌 2. Update these to match your training setup
    # EXPERIMENT_DIR = "ur5e_base_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_shoulder_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_forearm_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_upper_arm_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_wrist1_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_wrist2_link_model_latent"
    EXPERIMENT_DIR = "ur5e_wrist3_link_model_latent"

    # TARGET_LINK = "base"
    # TARGET_LINK = "shoulder_link"
    # TARGET_LINK = "forearm_link"
    # TARGET_LINK = "upper_arm_link"
    # TARGET_LINK = "wrist_1_link"
    # TARGET_LINK = "wrist_2_link"
    TARGET_LINK = "wrist_3_link"
    LATENT_SIZE = 256  # Must match the size used during training
    N_SAMPLES = 10000  # Increase for better visualization density

    EXPERIMENT_DIRS = [
        "ur5e_base_link_model_latent",
        "ur5e_shoulder_link_model_latent",
        "ur5e_forearm_link_model_latent",
        "ur5e_upper_arm_link_model_latent",
        "ur5e_wrist1_link_model_latent",
        "ur5e_wrist2_link_model_latent",
        "ur5e_wrist3_link_model_latent",
    ]

    TARGET_LINKS = [
        "base",
        "shoulder_link",
        "forearm_link",
        "upper_arm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    for index in range(len(TARGET_LINKS)):
        print(f"Starting DeepSDF Model Validation for link {TARGET_LINKS[index]}...")

        validate_model(
            robot_file=ROBOT_FILE_PATH,
            experiment_dir=EXPERIMENT_DIRS[index],
            target_link_name=TARGET_LINKS[index],
            latent_size=LATENT_SIZE,
            n_sample_points=N_SAMPLES,
        )
        print("Validation script finished.")
