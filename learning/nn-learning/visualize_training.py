from typing import final
import numpy as np
import trimesh
import torch
import os
import random
import sys
import scipy.io as sio  # For loading .mat files
from utils import (
    meshes_fk,
)

# --- Configuration ---
# You need to ensure the directory containing 'robot_sdf.py' is in your Python path
# If your files are in the same directory, this should work.
# If 'robot_sdf.py' imports from '.network_macros_mod', you might need to adjust imports
# or ensure a proper module structure if running this outside of the training setup.
# I'll assume `RobotSdfCollisionNet` is directly available or the path is set.
try:
    # Use the class name from your uploaded 'robot_sdf.py'
    from sdf.robot_sdf import RobotSdfCollisionNet

    # NOTE: Your robot_sdf.py imports from a relative path:
    # from .network_macros_mod import * # If the script fails, you might need to copy/mock the necessary network_macros_mod functions.
except ImportError as e:
    print(
        f"❌ CRITICAL ERROR: Could not import RobotSdfCollisionNet. Please check your module setup."
    )
    print(f"Details: {e}")
    sys.exit(1)

# Define the boundaries of the sampling cube (in meters)
MIN_BOUND = -0.1
MAX_BOUND = 0.1

# Define the resolution (number of samples along each axis)
# Warning: Resolution 60 -> 216,000 points. Resolution 100 -> 1,000,000 points.
RESOLUTION = 100

# Define the threshold for considering a point "on the surface"
SURFACE_THRESHOLD = 1.0  # meters (5 mm tolerance)
# Use a fixed number of samples, as random sampling density isn't uniform.
NUM_SAMPLES = 50
# Standard Deviation (spread) of the Gaussian distribution.
# This controls how far from the center (mean) the points are sampled.
# Since your bounds were [-1, 1], a std_dev of 0.3 to 0.5 is reasonable.
GAUSSIAN_STD_DEV = 0.5
GAUSSIAN_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Centered at the Original

# --- Random Seed for Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# --- End Configuration ---

# Define the network architecture parameters (taken from run_sdf.py and train_sdf.py)
NETWORK_LAYERS = [256] * 4  # (5 layers, with 1 removed because skips is empty)
NETWORK_SKIPS = []
OUTPUT_CHANNELS = 9  # Example: Assuming 7 links for a typical robot. ADJUST THIS!

# Define the path to your trained model weights (taken from run_sdf.py)
# MODEL_WEIGHTS_PATH = "sdf_256x5_mesh_py.pt"
MODEL_WEIGHTS_PATH = "franka_collision_model.pt"
# Define the path to your ground-truth mesh data
MESH_DATA_PATH = "../data-sampling/meshes/mesh_light_pts.mat"

# Define a fixed robot pose (joint angles) for which to visualize the SDF
# This needs to be a 1D NumPy array or list of the correct size (e.g., 7 joints for Franka)
FIXED_JOINT_COORDS_NP = np.array(
    [0.1, -1.5, 0.1, -1.5, 0.5, 0.6, 0.0], dtype=np.float32
)
# --- End Configuration ---


def load_network(model_path, joint_coord_size, output_size, device, layers, skips):
    """Loads the trained RobotSdfCollisionNet model based on your file structure."""

    in_channels = joint_coord_size + 3  # Joints + (x, y, z)
    print(f"🛠️ Initializing network: Input={in_channels}, Output={output_size}")

    # Initialize using the signature from your robot_sdf.py
    nn_model = RobotSdfCollisionNet(
        in_channels=in_channels, out_channels=output_size, layers=layers, skips=skips
    )

    tensor_args = {"device": device, "dtype": torch.float32}

    # Load weights using the method defined in your RobotSdfCollisionNet
    nn_model.load_weights(model_path, tensor_args)

    # Ensure model is on the correct device and in evaluation mode
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
    print(f"\n🧠 Starting surface sampling with resolution {res}^3...")

    # Create an array of sampling coordinates
    # coords = np.linspace(min_b, max_b, res)
    # X, Y, Z = np.meshgrid(coords, coords, coords)
    # points_3d = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    # Generate points from a normal distribution (Gaussian)
    # np.random.randn gives samples from a standard normal (mean=0, std=1)
    points_3d = np.random.randn(num_samples, 3).astype(np.float32)

    # Scale and shift the points to match the desired Gaussian:
    # Sample = (Standard Sample * Standard Deviation) + Mean
    points_3d = (points_3d * std_dev) + mean

    # Prepare inputs for the neural network
    num_points = points_3d.shape[0]

    # Broadcast the fixed joint coordinates across all points
    joints_repeated = np.tile(fixed_joints_np, (num_points, 1))

    # Combine [joints, point] -> Network input format
    input_data_np = np.hstack((joints_repeated, points_3d))

    # Convert to Tensor for the network
    input_tensor = torch.from_numpy(input_data_np).float().to(device)

    # --- Inference ---
    with torch.no_grad():
        # NOTE: Since your model is a composite class (RobotSdfCollisionNet has a 'model' attribute),
        # we call model.forward on the input_tensor. However, if the scaling/normalization
        # defined in compute_signed_distance is necessary, we must use that method.
        # Based on your run_sdf.py, the final call is `model.forward(x)`.

        # Check if model has normalization parameters (norm_dict)
        if hasattr(nn_model, "norm_dict"):
            # Use the method that handles scaling if it was trained with it
            # Your train_sdf.py shows mean_x/std_x are 0.0/1.0 ("disabled because of nerf features!")
            # but it is safer to check. We'll use the raw model.forward for maximum compatibility
            # with the inference path in run_sdf.py which uses raw tensors `y_pred = model.forward(x)`.
            # Since the goal is distance, we call forward on the core model
            output_distances = nn_model.model.forward(input_tensor)
            output_distances = output_distances

            # Apply scaling back to the output if norm_dict is present and needed
            if "y" in nn_model.norm_dict:
                std_y = nn_model.norm_dict["y"]["std"].to(device)
                mean_y = nn_model.norm_dict["y"]["mean"].to(device)
                # Your run_sdf.py has an extra 100* scaling on the input targets,
                # so the output must be scaled back by a factor of 100* and then by std_y/mean_y
                # For safety, let's assume the run_sdf.py path is correct:
                # y_pred = torch.mul(y_pred, std_y) + mean_y
                # To match run_sdf.py logic where scaling is mostly disabled:
                # output_distances = torch.mul(y_pred, std_y) + mean_y
                min_distances = output_distances.min(dim=1)[0].cpu().numpy()

            else:
                # If no scaling info, assume raw output is the distance
                min_distances = output_distances.min(dim=1)[0].cpu().numpy()
        else:
            # If no scaling info is available, call the underlying model directly.
            output_distances = nn_model.model.forward(input_tensor)
            min_distances = output_distances.min(dim=1)[0].cpu().numpy()

    positive_mask = min_distances > 0.0

    # Identify points where the min distance is close to zero
    surface_mask = min_distances < threshold
    # print(min_distances)
    # print(min(min_distances))
    for i in range(len(points_3d)):
        print(
            f"[{i}] point: {points_3d[i]}\n[{i}]dst: {min_distances[i]}\n[i]output:{output_distances[i].cpu().numpy()}"
        )

    # Extract the 3D coordinates that satisfy the surface condition
    surface_points = points_3d[surface_mask & positive_mask]
    surface_points = points_3d[
        (min_distances >= -threshold) & (min_distances <= threshold)
    ]
    filtered_dst = min_distances[
        (min_distances >= -threshold) & (min_distances <= threshold)
    ]

    print(f"✅ Finished sampling. Found {len(surface_points)} surface points.")
    return surface_points, filtered_dst


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    joint_coord_size = FIXED_JOINT_COORDS_NP.shape[0]

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

    # 2. Sample Points and Extract Surface
    surface_points = []
    surface_distances = []
    for _ in range(1):
        temp_points, temp_dst = get_surface_point_cloud(
            nn_model,
            FIXED_JOINT_COORDS_NP,
            MIN_BOUND,
            MAX_BOUND,
            RESOLUTION,
            SURFACE_THRESHOLD,
            NUM_SAMPLES,
            GAUSSIAN_STD_DEV,
            GAUSSIAN_MEAN,
            device,
        )
        for point_idx in range(len(temp_points)):
            surface_points.append(temp_points[point_idx])
            surface_distances.append(temp_dst[point_idx])
    surface_points = np.array(surface_points)
    # print(surface_distances)
    # Load .mat file
    try:
        mat_contents = sio.loadmat(MESH_DATA_PATH)
        mesh = mat_contents["mesh"][0]
    except FileNotFoundError:
        print(
            f"❌ ERROR: Mesh file not found at {MESH_DATA_PATH}. Cannot load ground truth."
        )
        return

    # Apply forward kinematics (using the placeholder/your actual function)
    base_transform = np.eye(4)
    # mesh_fk returns the list of transformed links
    transformed_link_meshes = meshes_fk(mesh, base_transform, FIXED_JOINT_COORDS_NP)
    meshes = []
    for j in range(OUTPUT_CHANNELS):
        V = transformed_link_meshes[j]["V"]  # Transformed Vertices
        F = transformed_link_meshes[j]["F"] - 1  # Faces
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=True)
        meshes.append(mesh)

    final_robot_mesh = trimesh.util.concatenate(meshes)
    # --- Visualization Setup ---
    scene = trimesh.Scene()

    # A. Add Ground-Truth Mesh (Original Robot Links)
    # We loop through the list of meshes returned by FK

    # Set the color and transparency (RGBA, A=0 to 255)
    final_robot_mesh.visual.face_colors = [
        50,
        50,
        200,
        180,
    ]  # Blueish, semi-transparent
    scene.add_geometry(final_robot_mesh)

    if surface_points.size == 0:
        print(
            "\n⚠️ No points found close to the surface. Try adjusting RESOLUTION or SURFACE_THRESHOLD."
        )
        return

    # 3. Use trimesh to create a point cloud visualization
    print("\n🖼️ Creating trimesh visualization...")

    # Create a trimesh point cloud object
    point_cloud = trimesh.points.PointCloud(surface_points)

    # Optional: Try to compute a mesh (Alpha Shape is better for surfaces than Convex Hull)
    try:
        # ball_radius is a heuristic for how dense the point cloud needs to be
        ball_radius = (MAX_BOUND - MIN_BOUND) / RESOLUTION * 1.5
        mesh = trimesh.points.PointCloud(surface_points).convex_hull
        print(
            "✅ Created convex hull mesh from surface points (may not be accurate for complex shapes)."
        )
        # scene.add_geometry(mesh)
        scene.add_geometry(point_cloud)
    except Exception as e:
        print(
            f"⚠️ Could not create a hull/mesh from the point cloud ({e}). Showing raw point cloud."
        )
        scene.add_geometry(point_cloud)

    # Add the sampling cube's bounds for context
    translation_transform = trimesh.transformations.translation_matrix([0, 0, 0])
    bounds = trimesh.creation.box(extents=[2, 2, 2], transform=translation_transform)
    bounds.visual.face_colors = [100, 100, 100, 50]  # Semi-transparent gray
    scene.add_geometry(bounds)

    # 4. Display the result
    print("🚀 Displaying visualization. Close the window to exit the script.")
    scene.show()


if __name__ == "__main__":
    main()
