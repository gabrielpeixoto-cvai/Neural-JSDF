from typing import final
import numpy as np
import trimesh
import torch
import time
import os
import random
import sys
import scipy.io as sio  # Required for original code, kept for reference
from utils import (
    meshes_fk,  # Kept for potential placeholder use, but replaced by mujoco_mesh_fk
)

# --- START OF NEW IMPORTS (Adapted from generate_dataset_mujoco.py) ---
import mujoco
from pathlib import Path
from typing import Tuple, List

# Define the path to your robot model file (adjust this path as needed)
# NOTE: This path is a placeholder from your generate_dataset_mujoco.py
ROBOT_FILE_PATH = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml"

# Placeholder functions from generate_dataset_mujoco.py (Need to be defined or imported)
# Since you didn't provide 'utils.py' or 'robot_sdf.py', I'm embedding the necessary
# MuJoCo functions from your provided scripts for a self-contained solution.


# --- Embedded MuJoCo Mesh Utility Functions ---
# (Copied from generate_dataset_mujoco.py / visualize_mujoco_sdf_data.py)
def get_geom_mesh_data_mjcf(model, geom_id):
    geom_type = model.geom_type[geom_id]
    geom_dataid = model.geom_dataid[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = geom_dataid
        vert_start = model.mesh_vertadr[mesh_id]
        vert_end = vert_start + model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_end = face_start + model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[vert_start:vert_end]
        faces = model.mesh_face[face_start:face_end]
        return vertices, faces
    else:
        return None, None


def get_body_mesh(model, body_id):
    meshes = []
    geom_id = model.body_geomadr[body_id]
    for i in range(model.body_geomnum[body_id]):
        gid = geom_id + i
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
    if not meshes:
        return None
    mesh = trimesh.util.concatenate(meshes)
    return mesh


def load_mujoco_model(robot_file_path_str: str) -> mujoco.MjModel:
    robot_file_path = Path(robot_file_path_str)
    try:
        robot_model = mujoco.MjModel.from_xml_path(str(robot_file_path))
        return robot_model
    except Exception as e:
        print(f"Error loading robot file: {e}")
        return None


def mujoco_mesh_fk(robot_model, qpos) -> List[trimesh.Trimesh]:
    d = mujoco.MjData(robot_model)
    d.qpos[:] = qpos
    mujoco.mj_forward(robot_model, d)
    mesh_data = []
    for body_id in range(robot_model.nbody):
        # The mesh loading inside the loop ensures that a fresh Trimesh object
        # is created for the link's base geometry before applying the FK transform.
        mesh = get_body_mesh(robot_model, body_id)
        if mesh is None:
            continue
        # Get the dynamic world transform components from mjData
        body_pos = d.xpos[body_id]
        body_mat = d.xmat[body_id].reshape(3, 3)
        # Construct the 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = body_mat
        T[:3, 3] = body_pos
        # Apply the transform
        mesh_world = mesh.copy()
        mesh_world.apply_transform(T)
        mesh_data.append(mesh_world)
    return mesh_data


# --- End of Embedded MuJoCo Mesh Utility Functions ---

# --- Configuration ---
try:
    from sdf.robot_sdf import RobotSdfCollisionNet
except ImportError as e:
    print(
        f"❌ CRITICAL ERROR: Could not import RobotSdfCollisionNet. Please check your module setup."
    )
    print(f"Details: {e}")
    sys.exit(1)

# Define the boundaries of the sampling cube (in meters)
MIN_BOUND = -1.5
MAX_BOUND = 1.5

# Define the resolution (number of samples along each axis)
RESOLUTION = 100

# Define the threshold for considering a point "on the surface"
SURFACE_THRESHOLD = 0.5
NUM_SAMPLES = 50000
GAUSSIAN_STD_DEV = 0.7
GAUSSIAN_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# --- Random Seed for Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# --- End Configuration ---

# Define the network architecture parameters
NETWORK_LAYERS = [256] * 4
NETWORK_SKIPS = []
# *** NOTE: OUTPUT_CHANNELS must match the number of links in your model. ***
# The UR5e model used in your other scripts typically has 9 meshes (links + base).
# I'll set it to 9 for compatibility with your npy_path/N_MESHES=9 in visualize_mujoco_sdf_data.py
OUTPUT_CHANNELS = 9

# Define the path to your trained model weights (taken from run_sdf.py)
MODEL_WEIGHTS_PATH = "ur5e_dataset_py_5000_sdf_256x5_mesh_py.pt"
# MESH_DATA_PATH is now unused, as we load the model directly via MuJoCo.
# MESH_DATA_PATH = "../data-sampling/meshes/mesh_light_pts.mat"

# Define a fixed robot pose (joint angles) for which to visualize the SDF
# NOTE: This must match the number of *actuated* joints in your MuJoCo model (UR5e has 6).
# The original code had 7, so you need to confirm the correct size. Assuming 6 for UR5e.
FIXED_JOINT_COORDS_NP = np.array(
    [0.1, -1.5, 0.1, -1.5, 0.5, 0.6], dtype=np.float32  # Reduced from 7 to 6 joints
)
# --- End Configuration ---


def load_network(model_path, joint_coord_size, output_size, device, layers, skips):
    """Loads the trained RobotSdfCollisionNet model."""
    in_channels = joint_coord_size + 3
    print(f"🛠️ Initializing network: Input={in_channels}, Output={output_size}")
    nn_model = RobotSdfCollisionNet(
        in_channels=in_channels, out_channels=output_size, layers=layers, skips=skips
    )
    tensor_args = {"device": device, "dtype": torch.float32}
    nn_model.load_weights(model_path, tensor_args)
    nn_model.model.to(**tensor_args)
    nn_model.model.eval()
    return nn_model


def get_surface_point_cloud(
    nn_model,
    fixed_joints_np,
    min_b,
    max_b,
    res,
    threshold,
    num_samples,
    std_dev,
    mean,
    device,
):
    """
    Samples the 3D space and extracts points near the SDF zero-level set.
    """
    print(f"\n🧠 Starting surface sampling...")

    points_3d = np.random.randn(num_samples, 3).astype(np.float32)
    points_3d = (points_3d * std_dev) + mean
    # points_3d = points_3d * 100

    num_points = points_3d.shape[0]
    joints_repeated = np.tile(fixed_joints_np, (num_points, 1))
    input_data_np = np.hstack((joints_repeated, points_3d))
    input_tensor = torch.from_numpy(input_data_np).float().to(device)

    # --- Inference ---
    with torch.no_grad():
        start = time.perf_counter()
        if hasattr(nn_model, "norm_dict"):
            output_distances = nn_model.model.forward(input_tensor)
            # Assuming a standard normalization path, otherwise this is a no-op
            if "y" in nn_model.norm_dict:
                std_y = nn_model.norm_dict["y"]["std"].to(device)
                mean_y = nn_model.norm_dict["y"]["mean"].to(device)
                # output_distances = torch.mul(output_distances, std_y) + mean_y
            min_distances = output_distances.min(dim=1)[0].cpu().numpy()
        else:
            output_distances = nn_model.model.forward(input_tensor)
            min_distances = output_distances.min(dim=1)[0].cpu().numpy()
        end = time.perf_counter()
    # print(points_3d)
    # print(output_distances)
    # print(min_distances)
    # print(max(min_distances))
    # for index in range(len(points_3d)):
    #    print(
    #        f"point: {points_3d[index]} distance: {output_distances[index].cpu().numpy()} min: {output_distances[index].min().cpu().numpy()}"
    #    )
    # Extract points close to the zero-level set (SDF surface)
    surface_points = points_3d[(min_distances >= 0.0) & (min_distances <= threshold)]
    filtered_dst = min_distances[(min_distances >= 0.0) & (min_distances <= threshold)]

    print(f"✅ Finished sampling. Found {len(surface_points)} surface points.")
    return surface_points, filtered_dst, end - start


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    joint_coord_size = FIXED_JOINT_COORDS_NP.shape[0]
    # --- START OF MODIFIED MESH LOADING/FK ---
    print("\n📦 Loading MuJoCo Model...")
    robot_model = load_mujoco_model(ROBOT_FILE_PATH)
    if robot_model is None:
        return

    # Compute FK and get the list of transformed Trimesh objects
    print("🤖 Computing Forward Kinematics for Fixed Joint Position...")
    transformed_link_meshes = mujoco_mesh_fk(robot_model, FIXED_JOINT_COORDS_NP)

    # Combine all individual link meshes into one final mesh for visualization
    if not transformed_link_meshes:
        print("❌ ERROR: No meshes were loaded from the MuJoCo model.")
        return

    OUTPUT_CHANNELS = len(transformed_link_meshes)

    # 1. Load the Network
    nn_model = load_network(
        MODEL_WEIGHTS_PATH,
        joint_coord_size,
        OUTPUT_CHANNELS,
        device,
        NETWORK_LAYERS,
        NETWORK_SKIPS,
    )
    if nn_model is None:
        return

    final_robot_mesh = trimesh.util.concatenate(transformed_link_meshes)
    # --- END OF MODIFIED MESH LOADING/FK ---

    # 2. Sample Points and Extract Surface
    # Original code samples 1000 times, which is computationally heavy.
    # We will loop once with a larger sample count to get a dense cloud.
    surface_points = []
    surface_distances = []
    times = []

    for _ in range(10):
        pt, dst, time = get_surface_point_cloud(
            nn_model,
            FIXED_JOINT_COORDS_NP,
            MIN_BOUND,
            MAX_BOUND,
            RESOLUTION,
            SURFACE_THRESHOLD,
            NUM_SAMPLES,  # Increased sample size for a better cloud
            GAUSSIAN_STD_DEV,
            GAUSSIAN_MEAN,
            device,
        )
        for point_idx in range(len(pt)):
            surface_points.append(pt[point_idx])
            surface_distances.append(dst[point_idx])
        times.append(time)
    # --- Visualization Setup ---
    surface_points = np.array(surface_points)
    # print(surface_points)
    # print(surface_distances)
    print(
        f"Total inference time for {NUM_SAMPLES*10} samples: {sum(times)} s average time for sample {sum(times)/len(surface_points)} s "
    )
    scene = trimesh.Scene()

    # A. Add Ground-Truth Mesh (Original Robot Links)
    # Set the color and transparency (RGBA, A=0 to 255)
    final_robot_mesh.visual.face_colors = [
        50,
        50,
        200,
        200,
    ]  # Blueish, semi-transparent
    scene.add_geometry(final_robot_mesh)

    if surface_points.size == 0:
        print(
            "\n⚠️ No points found close to the surface. Try adjusting RESOLUTION or SURFACE_THRESHOLD."
        )
        translation_transform = trimesh.transformations.translation_matrix([0, 0, 0])
        # Changed box extents to match bounds for better context
        bounds = trimesh.creation.box(
            extents=[MAX_BOUND - MIN_BOUND] * 3, transform=translation_transform
        )
        bounds.visual.face_colors = [100, 100, 100, 50]  # Semi-transparent gray
        scene.add_geometry(bounds)  # Commented out to reduce clutter

        # 4. Display the result
        print("🚀 Displaying visualization. Close the window to exit the script.")
        scene.show()

        return

    # 3. Use trimesh to create a point cloud visualization
    print("\n🖼️ Creating trimesh visualization...")

    # Create a trimesh point cloud object
    point_cloud = trimesh.points.PointCloud(surface_points)

    # Color the point cloud (optional, but good for visualization)
    point_cloud.colors = [0, 0, 0, 250]  # Set to Black

    scene.add_geometry(point_cloud)

    # Add the sampling cube's bounds for context
    translation_transform = trimesh.transformations.translation_matrix([0, 0, 0])
    # Changed box extents to match bounds for better context
    bounds = trimesh.creation.box(
        extents=[MAX_BOUND - MIN_BOUND] * 3, transform=translation_transform
    )
    bounds.visual.face_colors = [100, 100, 100, 50]  # Semi-transparent gray
    scene.add_geometry(bounds)  # Commented out to reduce clutter

    # 4. Display the result
    print("🚀 Displaying visualization. Close the window to exit the script.")
    scene.show()


if __name__ == "__main__":
    main()
