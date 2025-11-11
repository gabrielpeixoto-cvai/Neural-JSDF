import numpy as np
import trimesh
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import (
    get_bbox,
    scale_bbox,
    sample_bbox,
    meshes_fk,
    point_to_mesh_signed_distance,
)

import scipy.io as sio  # Required for .mat mesh loading
import mujoco
from pathlib import Path
from typing import Tuple, List
import trimesh


# from utils import meshes_fk # Required for Forward Kinematics
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
) -> Tuple[str, mujoco.MjModel]:
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
    for body_id in range(robot_model.nbody):
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mesh = get_body_mesh(robot_model, body_id)
        if mesh is None:
            continue
        mesh_data.append(mesh)

    return model_name, robot_model


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


def map_scalar_to_color(data, vmin=-0.1, vmax=0.1, cmap_name="coolwarm"):
    """Maps a 1D array of scalar data to an RGBA color array."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm(data))
    # Convert to 8-bit integer RGB/RGBA [0-255] as required by Trimesh
    colors_int = (colors * 255).astype(np.uint8)
    return colors_int


def map_scalar_to_color_custom(data, vmin=-0.1, vmax=0.1, zero_tolerance=1e-4):
    """
    Maps scalar data to Red (Negative), Black (Zero), and Blue (Positive) RGBA colors.

    Args:
        data (np.array): The d_min values.
        vmin (float): The negative cutoff point (e.g., -0.1).
        vmax (float): The positive cutoff point (e.g., 0.1).
        zero_tolerance (float): Defines the band around 0.0 to be considered "Black" (surface).

    Returns:
        np.array: A (N, 4) array of RGBA colors (0-255).
    """
    N = len(data)
    # Initialize all colors to Black (Surface) - RGBA (0, 0, 0, 255)
    colors = np.zeros((N, 4), dtype=np.uint8)
    colors[:, 3] = 255  # Set alpha channel to fully opaque

    # 1. Negative (Inside the mesh): Red
    # d_min < -tolerance: fully Red
    # Color: Red (255, 0, 0)
    idx_negative = data < -zero_tolerance
    colors[idx_negative, 0] = 255  # R

    # 2. Positive (Outside the mesh): Blue
    # d_min > +tolerance: fully Blue
    # Color: Blue (0, 0, 255)
    idx_positive = data > zero_tolerance
    colors[idx_positive, 2] = 255  # B

    # 3. Surface (Near Zero): Black
    # -tolerance <= d_min <= +tolerance: remains Black (0, 0, 0)
    data_zero = data[(data >= -zero_tolerance) & (data <= zero_tolerance)]
    print(f"ZEROS: {data_zero}")

    # You can also add gradients for values that are transitioning from the full color:
    # Example: Fading from full Red at vmin to Black at -tolerance,
    #          and Black at +tolerance to full Blue at vmax.

    # Gradient for Deep Inside (between vmin and -tolerance) - optional
    # idx_deep_neg = (data >= vmin) & (data < -zero_tolerance)
    # Scale from 0.0 at -tolerance to 1.0 at vmin (more negative)
    # scale_neg = 1.0 - (data[idx_deep_neg] + zero_tolerance) / (vmin + zero_tolerance)
    # colors[idx_deep_neg, 0] = (scale_neg * 255).astype(np.uint8)

    # Gradient for Far Outside (between +tolerance and vmax) - optional
    # idx_far_pos = (data > zero_tolerance) & (data <= vmax)
    # Scale from 0.0 at +tolerance to 1.0 at vmax (more positive)
    # scale_pos = (data[idx_far_pos] - zero_tolerance) / (vmax - zero_tolerance)
    # colors[idx_far_pos, 2] = (scale_pos * 255).astype(np.uint8)

    return colors


def visualize_sdf_points_trimesh(
    npy_path="ur5e_dataset_py.npy", N_MESHES=9, N_PTS_PER_JPOS=990
):
    """
    Loads the dataset, selects a single joint position's data,
    and visualizes the 3D points colored by d_min alongside the
    transformed robot links using Trimesh.
    """
    try:
        dataset = np.load(npy_path)
    except FileNotFoundError:
        print(f"ERROR: '{npy_path}' not found. Cannot visualize dataset.")
        return

    # 1. Extract and Filter Data for the First Joint Position
    print(dataset.shape)
    print(dataset[0])
    pts_all = dataset[:, 6:9]
    dist_arr_all = 100 * dataset[:, 9:]
    q = dataset[0, 0:6]  # First joint position
    robot_name, robot_model = load_data()
    limits = get_joint_limits(robot_model)
    meshes = mujoco_mesh_fk(robot_model, q)
    N_PTS_PER_JPOS = 110 * len(meshes)

    pts_plot = pts_all[:N_PTS_PER_JPOS]
    dist_arr = dist_arr_all[:N_PTS_PER_JPOS]
    d_min_plot = np.min(dist_arr, axis=1)

    print(f"Plotting for joint position (q): {q}")
    print(f"Plotting {pts_plot.shape[0]} query points.")
    print(f"Plotting {dist_arr.shape} query points.")
    print(f"DMIN: {d_min_plot}")

    # 2. Color the Query Points by d_min
    # Use a range centered around 0 (the surface)
    vmax = np.max(d_min_plot)
    vmin = np.min(d_min_plot)
    point_colors = map_scalar_to_color_custom(d_min_plot, vmin=vmin, vmax=vmax)

    # Create Trimesh Point Cloud for SDF data
    sdf_point_cloud = trimesh.points.PointCloud(pts_plot, colors=point_colors)

    # 3. Load and Transform Meshes
    scene_entities = [sdf_point_cloud]
    N_MESHES = len(meshes)
    if meshes is not None:
        mesh_fk = mujoco_mesh_fk(robot_model, q)

        if mesh_fk is not None:
            for j in range(N_MESHES):
                V = mesh_fk[j].vertices
                F = mesh_fk[j].faces

                link_mesh = trimesh.Trimesh(V, F)
                # Set a translucent gray color for the mesh (RGBA 0-255)
                link_mesh.visual.face_colors = [150, 150, 150, 150]
                scene_entities.append(link_mesh)

    # 4. Create and Show Scene
    scene = trimesh.Scene(scene_entities)

    # Use scene.show() for an interactive window, or scene.save() to write a file
    # scene.save('sdf_visualization_trimesh_scene.gltf')
    print("\nAttempting to launch Trimesh viewer (requires a suitable backend).")
    scene.show()


def diagnose_dmin_range(npy_path="robot_dataset_py.npy", N_PTS_PER_JPOS=990):
    """Calculates and prints statistics for the subset of d_min values being plotted."""
    try:
        dataset = np.load(npy_path)
    except FileNotFoundError:
        print(f"ERROR: '{npy_path}' not found. Cannot diagnose dataset.")
        return

    # Extract data for the first joint position
    # dist_arr_all starts at index 10
    dist_arr_all = 100 * dataset[:, 10:]

    # Get subset for the first joint configuration
    dist_arr = dist_arr_all[:N_PTS_PER_JPOS]
    # Calculate minimum signed distance across all links for each point
    d_min_plot = np.min(dist_arr, axis=1)

    print("\n--- Minimum Signed Distance (d_min) Statistics ---")
    print(f"Total points analyzed: {d_min_plot.shape[0]}")
    print(f"  Minimum d_min: {np.min(d_min_plot):.4f} (Should be negative/zero)")
    print(
        f"  Maximum d_min: {np.max(d_min_plot):.4f} (This is the largest distance found)"
    )
    print(f"  Mean d_min: {np.mean(d_min_plot):.4f}")
    print("--------------------------------------------------")


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

    print(f"MIN: {joint_limit_min}\nMAX:{joint_limit_max}")
    return np.array(joint_limit_min), np.array(joint_limit_max)


diagnose_dmin_range()
# Execute the visualization function
# visualize_sdf_points_and_meshes(N_MESHES=9)
visualize_sdf_points_trimesh(N_MESHES=9)
