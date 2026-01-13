import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import scipy.io as sio  # For loading .mat files
from utils import (
    get_bbox,
    scale_bbox,
    sample_bbox,
    meshes_fk,
    point_to_mesh_signed_distance,
)
import trimesh
import mujoco
from typing import Tuple, List, Dict
import igl


# --- Utility Functions from Original Script (Included for completeness) ---

# NOTE: Keeping get_geom_mesh_data_mjcf, get_body_mesh, load_data,
# mujoco_mesh_fk, get_joint_limits, map_scalar_to_color_custom,
# display_mujoco_meshes_and_sdfs as they are useful for initial setup
# and the final visualization step.

DEBUG = False


def get_geom_mesh_data_mjcf(model, geom_id):
    """
    Extracts mesh vertices and faces for a given geom ID from a Mujoco model.
    """
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
    """
    Collects and merges all mesh geoms belonging to a body into a single Trimesh.
    """
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


def load_data(
    robot_file_path_str: str = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml",
) -> Tuple[str, mujoco.MjModel, List[trimesh.Trimesh], List[str]]:
    # 📌 UPDATE THIS PATH TO YOUR ROBOT'S URDF or MJCF FILE.
    robot_file_path = Path(robot_file_path_str)
    model_name = robot_file_path_str.split("/")[-1].split(".")[0]
    robot_dir = robot_file_path.parent

    try:
        robot_model = mujoco.MjModel.from_xml_path(str(robot_file_path))
    except Exception as e:
        print(f"Error loading robot file: {e}")
        exit()

    mesh_data = []
    mesh_names = []
    for body_id in range(robot_model.nbody):
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mesh = get_body_mesh(robot_model, body_id)
        if mesh is None:
            continue
        mesh_data.append(mesh)
        mesh_names.append(body_name)

    return model_name, robot_model, mesh_data, mesh_names


def mujoco_mesh_fk(robot_model, qpos) -> List[trimesh.Trimesh]:
    d = mujoco.MjData(robot_model)
    d.qpos[:] = qpos
    mujoco.mj_forward(robot_model, d)

    mesh_data = []
    for body_id in range(robot_model.nbody):
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mesh = get_body_mesh(robot_model, body_id)
        if mesh is None:
            continue
        # 2. Get the dynamic world transform components from mjData
        body_pos = d.xpos[body_id]
        body_mat = d.xmat[body_id].reshape(3, 3)

        # 3. Construct the 4x4 homogeneous transformation matrix
        T = np.eye(4)
        # Rotation (top-left 3x3)
        T[:3, :3] = body_mat
        # Translation (top-right 3x1)
        T[:3, 3] = body_pos

        # 4. Create a copy of the mesh and apply the transform directly
        mesh_world = mesh.copy()
        mesh_world.apply_transform(T)
        mesh_data.append(mesh_world)
    return mesh_data


def get_joint_limits(m: mujoco.MjModel):
    """
    Extracts the joint limits (min and max position) for all joints
    defined with limits in the MuJoCo model.
    """
    joint_limit_min = []
    joint_limit_max = []

    # Iterate through all joints in the model
    for joint_id in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        # Check if the joint has limits defined (m.jnt_limited)
        # 1 means limited, 0 means unlimited
        if m.jnt_limited[joint_id] == 1:
            # The limits are stored in m.jnt_range[joint_id]
            # This array contains [min_limit, max_limit]

            qpos_range = m.jnt_range[joint_id]

            joint_limit_min.append(qpos_range[0])
            joint_limit_max.append(qpos_range[1])

    return np.array(joint_limit_min), np.array(joint_limit_max)


def display_mujoco_meshes(transformed_meshes: List[trimesh.Trimesh]):
    """
    Combines and displays a dictionary of transformed trimesh objects
    in an interactive viewer.
    """
    if not transformed_meshes:
        print("No meshes to display.")
        return

    scene = trimesh.Scene()
    for mesh in transformed_meshes:
        if mesh.visual.kind == "face":
            color = np.random.randint(0, 256, 4)  # RGBA color
            color[3] = 200  # Set alpha for visibility
            mesh.visual.face_colors = color
        scene.add_geometry(mesh)
        scene.show()

    print(f"Displaying {len(transformed_meshes)} meshes in the viewer...")
    scene.show()


def map_scalar_to_color_custom(data, zero_tolerance=1e-4):
    """
    Maps scalar data to Red (Negative/Inside), Black (Zero/Surface), and Blue (Positive/Outside) RGBA colors.
    """
    N = len(data)
    colors = np.zeros((N, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Fully opaque

    # 1. Negative (Inside the mesh): Red
    idx_negative = data < -zero_tolerance
    colors[idx_negative, 0] = 255  # R

    # 2. Positive (Outside the mesh): Blue
    idx_positive = data > zero_tolerance
    colors[idx_positive, 2] = 255  # B

    # 3. Surface (Near Zero): Black remains

    return colors


def display_mujoco_meshes_and_sdfs(
    transformed_meshes: List[trimesh.Trimesh],
    points_world: np.ndarray,
    sdfs: np.ndarray,
    n_pts: int,
    point_size: float = 3.0,  # Added parameter for point size
):
    """
    Combines and displays the transformed robot meshes and the sampled SDF points
    in the world frame using Trimesh's viewer.

    Args:
        transformed_meshes (List[trimesh.Trimesh]): Meshes transformed to the world frame.
        points_world (np.ndarray): All sampled points in world coordinates (N_total, 3).
        sdfs (np.ndarray): The minimum signed distance for each point (N_total,).
    """
    # Use the minimum SDF distance over all links for visualization color
    point_colors = map_scalar_to_color_custom(sdfs)

    # Create Trimesh Point Cloud for SDF data
    sdf_point_cloud = trimesh.points.PointCloud(
        points_world,
        colors=point_colors,
        metadata={"point_size": point_size},  # Trimesh uses metadata for point size
    )

    if not transformed_meshes:
        print("No meshes to display.")
        return

    # 1. Create a Scene object and add the point cloud
    # --- 2. Scene Composition (Step-by-Step Addition) ---
    scene = trimesh.Scene()

    # Add the Point Cloud
    scene.add_geometry(sdf_point_cloud)

    # 2. Add each transformed mesh to the scene
    meshes = []
    for i, mesh in enumerate(transformed_meshes):
        # Gray and slightly transparent for the robot links
        # if mesh.visual.kind == "face":
        color = np.array([150, 150, 150, 150])
        mesh.visual.face_colors = color
        # IMPORTANT FIX: Add geometry explicitly with smooth=False and process=False
        # This prevents Trimesh from running internal, slow, and error-prone checks
        # like computing smooth_shaded and submeshes, which caused the crash.
        meshes.append(mesh)
    final_mesh = trimesh.util.concatenate(meshes)

    scene.add_geometry(final_mesh)
    # scene.show()

    print(
        f"\nDisplaying {len(transformed_meshes)} meshes and {n_pts} points in the viewer..."
    )
    print("SDF Color Key: Red=Inside, Black=Surface, Blue=Outside")

    # 4. Show the scene (this is a blocking call until the viewer is closed)
    # scene.show()
    try:
        # 4. Show the scene (this is a blocking call)
        scene.show(process=False)
    except Exception as e:
        print(f"\n[ERROR] Trimesh viewer failed to start or render.")
        print(
            f"Possible causes: Missing dependencies (like pyglet), no OpenGL context (remote connection), or environment issues."
        )
        print(f"Error details: {e}")


# --- New Local Dataset Generation Function ---


def generate_local_dataset(
    mesh_data: List[trimesh.Trimesh], mesh_names: List[str], output_dir: Path
) -> None:
    """
    Generates a dataset of [x, y, z, signed_distance] for each link mesh
    in its local coordinate frame.

    Args:
        mesh_data (List[trimesh.Trimesh]): List of untransformed link meshes.
        mesh_names (List[str]): Names corresponding to the meshes.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are link names and
                              values are the local SDF dataset (N_samples, 4).
    """
    # 2. Dataset Configuration (Same ratios as original for consistency)
    N_MESHES = len(mesh_data)
    N_SAMPLES_LINK = 10000  # Total samples per link (adjust as needed)

    # Ratios for point types
    SAMPLE_RATIO_INSIDE = 0.25
    SAMPLE_RATIO_OUTSIDE = 0.35
    SAMPLE_RATIO_CLOSE = 0.15
    SAMPLE_RATIO_FAR = 0.15
    SAMPLE_RATIO_ZERO = 0.10

    # Points per mesh per type - scaled to the total number of samples
    N_INSIDE = int(N_SAMPLES_LINK * SAMPLE_RATIO_INSIDE)
    N_OUTSIDE = int(N_SAMPLES_LINK * SAMPLE_RATIO_OUTSIDE)
    N_CLOSE = int(N_SAMPLES_LINK * SAMPLE_RATIO_CLOSE)
    N_FAR = int(N_SAMPLES_LINK * SAMPLE_RATIO_FAR)
    N_ZERO = int(N_SAMPLES_LINK * SAMPLE_RATIO_ZERO)

    # Ensure the sum is correct, adjust N_ZERO if necessary due to rounding
    N_TOTAL_SAMPLED = N_INSIDE + N_OUTSIDE + N_CLOSE + N_FAR + N_ZERO
    if N_TOTAL_SAMPLED != N_SAMPLES_LINK:
        N_ZERO += N_SAMPLES_LINK - N_TOTAL_SAMPLED

    box_delta = 0.01
    # Global Bounding box for 'far' points. Assuming a roughly centered robot at world origin (0,0,0)
    bbox_far_global = {
        "xmin": -1,
        "xmax": 1,
        "ymin": -1,
        "ymax": 1,
        "zmin": -1,
        "zmax": 1,
    }

    print(
        f"\n--- Generating Local SDF Dataset for {N_MESHES} Links ({N_SAMPLES_LINK} points each) ---"
    )
    total_samples = 0

    for j in tqdm(range(N_MESHES), desc="Generating link datasets"):
        mesh = mesh_data[j]
        link_name = mesh_names[j]
        V = mesh.vertices
        F = mesh.faces

        # Get the bounding box for the *untransformed* mesh (local frame)
        bbox = get_bbox(V)

        # --- Sample Points in Local Frame ---

        # 1. Points INSIDE link (by scaling the bbox inwards)
        bbox_inside = scale_bbox(bbox, -box_delta)
        pts_inside = sample_bbox(bbox_inside, N_INSIDE)

        # 2. Points OUTSIDE link (by scaling the bbox outwards)
        bbox_outside = scale_bbox(bbox, box_delta)
        pts_outside_large = sample_bbox(bbox_outside, N_OUTSIDE)

        # Filter to keep only points outside the *original* bbox
        outside_filter = (
            (pts_outside_large[:, 0] < bbox["xmin"])
            | (pts_outside_large[:, 0] > bbox["xmax"])
            | (pts_outside_large[:, 1] < bbox["ymin"])
            | (pts_outside_large[:, 1] > bbox["ymax"])
            | (pts_outside_large[:, 2] < bbox["zmin"])
            | (pts_outside_large[:, 2] > bbox["zmax"])
        )
        pts_outside = pts_outside_large[outside_filter]

        # 3. Points CLOSE to link (samples in the slightly expanded bbox)
        bbox_close = scale_bbox(bbox, box_delta)
        pts_close = sample_bbox(bbox_close, N_CLOSE)

        # 4. Points FAR from link (global bbox - this is a bit arbitrary in local frame, but maintains the density)
        # We can use the global bbox here as a dense sampler over a large volume, relative to the link's origin
        pts_far = sample_bbox(bbox_far_global, N_FAR)

        # 5. Points ON link surface (sampled from mesh vertices)
        rand_indices = np.random.randint(0, len(V), N_ZERO)
        pts_mesh = V[rand_indices, :]

        # Combine all sampled points for this link
        pts_link_local = np.vstack(
            [pts_inside, pts_outside, pts_close, pts_far, pts_mesh]
        )
        n_pts_link = pts_link_local.shape[0]

        # 4. Distance Calculation (still in local frame)
        # point_to_mesh_signed_distance handles the 1-indexing conversion internally
        signed_distances, trimesh_mesh = point_to_mesh_signed_distance(
            F, V, pts_link_local
        )

        # necessary to handle non-watertight meshes
        # ray-casting method
        # points_inside = trimesh_mesh.contains(pts_link_local)

        # WITH LIBIGL WINDING NUMBER:
        # igl expects vertices and faces as double/int numpy arrays
        # Note: F must be (N, 3) int32/64, V must be (N, 3) float64
        winding_numbers = igl.winding_number(
            V.astype(np.float64), F.astype(np.int64), pts_link_local.astype(np.float64)
        )

        # A winding number > 0.5 typically indicates the point is inside
        points_inside = winding_numbers > 0.5

        signed_distances_abs = abs(signed_distances)
        signed_distances_abs[np.where(points_inside)] = (
            -1 * signed_distances_abs[np.where(points_inside)]
        )

        # 5. Create and Store Dataset Entry: [x, y, z, d]
        local_dataset = np.hstack([pts_link_local, signed_distances_abs[:, np.newaxis]])

        display_local_sampling(link_name, trimesh_mesh, local_dataset)
        # Save each link's dataset to a separate file
        save_path = output_dir / f"{link_name}_sdf.npy"
        np.save(save_path, local_dataset)
        print(
            f"Saved dataset for {link_name} with {local_dataset.shape[0]} samples to {save_path.name}"
        )
        total_samples += local_dataset.shape[0]

    print(
        f"\nLocal Dataset generation complete. Total samples across all links: {total_samples}"
    )
    print(f"Data saved to directory '{output_dir}'. Columns: [x, y, z, d]")


# --- NEW Local Visualization Function ---


def display_local_sampling(
    link_name: str,
    local_mesh: trimesh.Trimesh,
    dataset: np.ndarray,
    point_size: float = 5.0,
):
    """
    Visualizes a single link's mesh and its sampled SDF points in its local frame.
    """
    pts_local = dataset[:, :3]
    sdfs_local = dataset[:, 3]

    print(f"\n--- Visualizing Local Sampling for Link: {link_name} ---")
    print(f"Total points: {pts_local.shape[0]}.")
    print("SDF Color Key: Red=Inside, Black=Surface, Blue=Outside")

    # 1. Color the points based on SDF
    point_colors = map_scalar_to_color_custom(sdfs_local)

    # 2. Create Point Cloud
    sdf_point_cloud = trimesh.points.PointCloud(
        pts_local, colors=point_colors, metadata={"point_size": point_size}
    )

    # 3. Setup Scene
    scene = trimesh.Scene()
    scene.add_geometry(sdf_point_cloud)

    # 4. Add the local mesh
    # if local_mesh.visual.kind == "face":
    color = np.array([150, 150, 150, 150])
    local_mesh.visual.face_colors = color

    scene.add_geometry(local_mesh)

    try:
        scene.show(smooth=False)
    except Exception as e:
        print(f"\n[ERROR] Local Trimesh viewer failed: {e}")


# --- Main Dataset Generation and Visualization Script ---

if __name__ == "__main__":

    # 1. Setup and Initialization
    # Load model and get *untransformed* (local) meshes and their names
    try:
        # NOTE: load_data is updated to return mesh_names
        model_name, robot_model, local_mesh_data, mesh_names = load_data()
    except Exception as exc:
        print(f"Error loading model: {exc}")
        exit()

    q_min_raw, q_max_raw = get_joint_limits(robot_model)
    # display_mujoco_meshes(local_mesh_data)

    # 3. Save the Datasets (E.g., one .npy file per link)
    output_dir = Path(f"{model_name}_local_sdf_datasets")
    output_dir.mkdir(exist_ok=True)

    # 2. Generate Local SDF Datasets
    generate_local_dataset(local_mesh_data, mesh_names, output_dir)

    # --- 4. Visualization Step (using a random joint angle) ---

    print("\n--- Starting Visualization ---")

    # Randomly sample a joint position (q) for visualization
    q_span = q_max_raw - q_min_raw
    # Inflate limits by 5% as was done in original script setup
    q_min = q_min_raw - q_span * 0.05
    q_max = q_max_raw + q_span * 0.05

    q_rand = q_min + np.random.rand(1, len(q_min)) * (q_max - q_min)
    jpos_vis = [0, 0, 0, 0, 0, 0]  # Joint position vector

    # Compute FK to get transformed meshes (world frame)
    transformed_meshes = mujoco_mesh_fk(robot_model, jpos_vis)
    # display_mujoco_meshes(transformed_meshes)

    # Combine all points into a single world-frame point cloud
    all_points_world = np.empty((0, 3))
    all_sdfs = np.empty((0,))

    # Get the world transformation for each link
    d = mujoco.MjData(robot_model)
    d.qpos[:] = jpos_vis
    mujoco.mj_forward(robot_model, d)

    print(f"Visualizing with joint angles: {jpos_vis}")

    # Reconstruct the world point cloud from the local datasets
    for body_id in range(robot_model.nbody):
        # 1. Get the transformation T_world_link for this joint configuration
        link_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if link_name not in mesh_names:
            continue
        body_pos = d.xpos[body_id]
        body_mat = d.xmat[body_id].reshape(3, 3)

        T_world_link = np.eye(4)
        T_world_link[:3, :3] = body_mat
        T_world_link[:3, 3] = body_pos

        data_path = output_dir / f"{link_name}_sdf.npy"
        local_dataset = np.load(data_path)
        # 2. Get the local dataset [x, y, z, d]
        dataset = local_dataset
        pts_local = dataset[:, :3]
        sdfs_local = dataset[:, 3]

        # 3. Transform points from local frame (L) to world frame (W)
        # P_W = T_W_L @ P_L
        pts_local_h = np.hstack(
            [pts_local, np.ones((pts_local.shape[0], 1))]
        ).T  # (4, N)
        pts_world_h = T_world_link @ pts_local_h
        pts_world = pts_world_h[:3, :].T  # (N, 3)

        # 4. Append to global lists
        all_points_world = np.vstack([all_points_world, pts_world])
        all_sdfs = np.hstack([all_sdfs, sdfs_local])

    # Display the transformed meshes and the colored point cloud
    display_mujoco_meshes_and_sdfs(
        transformed_meshes, all_points_world, all_sdfs, all_points_world.shape[0]
    )
