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
from typing import Tuple, List


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

        # print(f"POS: {body_pos}")
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

    Args:
        transformed_meshes (Dict[str, trimesh.Trimesh]): A dictionary
            where values are trimesh objects transformed to the world frame.
    """
    if not transformed_meshes:
        print("No meshes to display.")
        return

    # 1. Create a Scene object
    scene = trimesh.Scene()

    # 2. Add each transformed mesh to the scene
    for mesh in transformed_meshes:
        # Optionally, set a random color for better visual separation
        if mesh.visual.kind == "face":
            # Generate a random color for the body if it doesn't have one
            color = np.random.randint(0, 256, 4)  # RGBA color
            color[3] = 200  # Set alpha for visibility
            mesh.visual.face_colors = color

        # Add the mesh to the scene
        scene.add_geometry(mesh)

    # 3. Add a basic ground plane for context (optional but helpful)
    # ground = trimesh.creation.box(extents=[5, 5, 0.01], transform=trimesh.transformations.translation_matrix([0, 0, -0.005]))
    # scene.add_geometry(ground, name='ground', geom_name='ground_geom')

    print(f"Displaying {len(transformed_meshes)} meshes in the viewer...")

    # 4. Show the scene (this is a blocking call until the viewer is closed)
    scene.show()


def map_scalar_to_color_custom(data, zero_tolerance=1e-4):
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


def display_mujoco_meshes_and_sdfs(
    transformed_meshes: List[trimesh.Trimesh], points: np.ndarray, sdfs: np.ndarray
):
    """
    Combines and displays a dictionary of transformed trimesh objects
    in an interactive viewer.

    Args:
        transformed_meshes (Dict[str, trimesh.Trimesh]): A dictionary
            where values are trimesh objects transformed to the world frame.
    """

    d_min_plot = np.min(dist_arr, axis=1)
    point_colors = map_scalar_to_color_custom(d_min_plot)

    # Create Trimesh Point Cloud for SDF data
    sdf_point_cloud = trimesh.points.PointCloud(points, colors=point_colors)

    if not transformed_meshes:
        print("No meshes to display.")
        return

    # 1. Create a Scene object
    scene_entities = [sdf_point_cloud]

    # 2. Add each transformed mesh to the scene
    for mesh in transformed_meshes:
        # Optionally, set a random color for better visual separation
        if mesh.visual.kind == "face":
            # Generate a random color for the body if it doesn't have one
            color = np.array([150, 150, 150, 150])
            mesh.visual.face_colors = color

        # Add the mesh to the scene
        scene_entities.append(mesh)

    scene = trimesh.Scene(scene_entities)
    # 3. Add a basic ground plane for context (optional but helpful)
    # ground = trimesh.creation.box(extents=[5, 5, 0.01], transform=trimesh.transformations.translation_matrix([0, 0, -0.005]))
    # scene.add_geometry(ground, name='ground', geom_name='ground_geom')

    print(f"Displaying {len(transformed_meshes)} meshes in the viewer...")

    # 4. Show the scene (this is a blocking call until the viewer is closed)
    scene.show()


# --- Main Dataset Generation Script ---

if __name__ == "__main__":

    # 1. Setup and Initialization

    # Load mesh data - Adjust path if necessary.
    try:
        model_name, robot_model = load_data()
    except Exception as exc:
        print(f"Error loading model: {exc}")
        exit()

    q_min_raw, q_max_raw = get_joint_limits(robot_model)

    zero_q = np.zeros(q_min_raw.shape)

    # Inflate limits by 5% as per MATLAB code
    q_span = q_max_raw - q_min_raw
    q_min = q_min_raw - q_span * 0.05
    q_max = q_max_raw + q_span * 0.05
    #
    mesh_data = mujoco_mesh_fk(robot_model, zero_q)

    # 2. Dataset Configuration
    N_MESHES = len(mesh_data)
    # N_JPOS = 10  # Number of joint positions to sample
    N_JPOS = 500  # Uncomment for the value used in the paper
    # N_SAMPLES_JPOS = 110 # original
    N_SAMPLES_JPOS = 220  # original

    SAMPLE_RATIO_INSIDE = 0.25
    SAMPLE_RATIO_OUTSIDE = 0.35
    SAMPLE_RATIO_CLOSE = 0.15  # original 20
    SAMPLE_RATIO_FAR = 0.15  # original 20
    SAMPLE_RATIO_ZERO = 0.10

    # Points per mesh per type (from genDataset.m)
    # N_INSIDE = np.full(N_MESHES, 25)
    # N_OUTSIDE = np.full(N_MESHES, 35)
    # N_CLOSE = np.full(N_MESHES, 20)
    # N_FAR = np.full(N_MESHES, 20)
    # N_ZERO = np.full(N_MESHES, 10)
    N_INSIDE = np.full(N_MESHES, int(N_SAMPLES_JPOS / SAMPLE_RATIO_INSIDE))
    N_OUTSIDE = np.full(N_MESHES, int(N_SAMPLES_JPOS / SAMPLE_RATIO_OUTSIDE))
    N_CLOSE = np.full(N_MESHES, int(N_SAMPLES_JPOS / SAMPLE_RATIO_CLOSE))
    N_FAR = np.full(N_MESHES, int(N_SAMPLES_JPOS / SAMPLE_RATIO_FAR))
    N_ZERO = np.full(N_MESHES, int(N_SAMPLES_JPOS / SAMPLE_RATIO_ZERO))

    box_delta = 0.1
    # Bounding box for 'far' points (global workspace)
    bbox_far = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "zmin": -1, "zmax": 1}
    #

    all_data = []

    # 3. Main Data Generation Loop
    for i in tqdm(range(N_JPOS)):
        # print(f"Generating data for joint position {i+1}/{N_JPOS}...")

        # Sample a random joint position (q)
        q_rand = q_min + np.random.rand(1, len(q_min)) * (q_max - q_min)
        jpos = q_rand[0]  # Joint position vector

        # Compute FK and transform meshes
        mesh_fk = mujoco_mesh_fk(robot_model, jpos)
        # if DEBUG:
        #    display_mujoco_meshes(mesh_fk)

        pts_all = np.empty((0, 3))

        # Iterate through each relevant robot link mesh
        for j in range(N_MESHES):
            V = mesh_fk[j].vertices
            F = mesh_fk[j].faces

            # Get the bounding box for the *transformed* mesh
            bbox = get_bbox(V)

            # --- Sample Points ---

            # 1. Points INSIDE link (by scaling the bbox inwards)
            bbox_inside = scale_bbox(bbox, -box_delta)
            pts_inside = sample_bbox(bbox_inside, N_INSIDE[j])

            # 2. Points OUTSIDE link (by scaling the bbox outwards)
            bbox_outside = scale_bbox(bbox, box_delta)
            pts_outside_large = sample_bbox(bbox_outside, N_OUTSIDE[j])

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
            # The MATLAB code samples in the original bbox region (scaleBBox(bbox, -box_delta) then scaleBBox(..., box_delta) is basically bbox)
            bbox_close = scale_bbox(bbox, box_delta)
            pts_close = sample_bbox(bbox_close, N_CLOSE[j])

            # 4. Points FAR from link (global bbox)
            pts_far = sample_bbox(bbox_far, N_FAR[j])

            # 5. Points ON link surface (sampled from mesh vertices)
            rand_indices = np.random.randint(0, len(V), N_ZERO[j])
            pts_mesh = V[rand_indices, :]

            # Combine all sampled points for this link
            pts_link = np.vstack(
                [pts_inside, pts_outside, pts_close, pts_far, pts_mesh]
            )
            pts_all = np.vstack([pts_all, pts_link])
            #

        # 4. Distance Calculation
        n_pts = pts_all.shape[0]
        dist_arr = np.zeros((n_pts, N_MESHES))

        # MATLAB uses point2trimesh followed by inpolyhedron to determine the sign.
        # Python uses trimesh.proximity.signed_distance for a single, robust call.
        for j in range(N_MESHES):
            V = mesh_fk[j].vertices
            F = mesh_fk[j].faces

            # point_to_mesh_signed_distance handles the 1-indexing conversion internally
            signed_distances, trimesh_mesh = point_to_mesh_signed_distance(
                F, V, pts_all
            )
            # necessary to handle non-watertight meshes
            points_inside = trimesh_mesh.contains(pts_all)
            signed_distances_abs = abs(signed_distances)
            signed_distances_abs[np.where(points_inside)] = (
                -1 * signed_distances_abs[np.where(points_inside)]
            )
            lbl_inside = np.where(signed_distances_abs < 0)[0]

            dist_arr[:, j] = signed_distances_abs
        #

        # 5. Create and Store Dataset Entry
        # The joint state (jpos) is 8 elements. We only use the first 7 for the dataset.
        joint_cols = np.tile(jpos, (n_pts, 1))

        # subdataset = [q(1:7), pts_all (x,y,z), dist_arr (d1...dN)]
        subdataset = np.hstack([joint_cols, pts_all, dist_arr])
        if DEBUG:
            display_mujoco_meshes_and_sdfs(mesh_fk, pts_all, dist_arr)
        all_data.append(subdataset)

    # Final Concatenation and Save
    final_dataset = np.vstack(all_data)

    # Save the dataset to a file (e.g., NumPy .npy or CSV)
    np.save(f"{model_name}_dataset_py_{N_JPOS}_{N_SAMPLES_JPOS}.npy", final_dataset)
    # np.savetxt('robot_dataset.csv', final_dataset, delimiter=',')

    print(f"\nDataset generation complete. Total samples: {final_dataset.shape[0]}")
    print(
        f"Data saved to 'robot_dataset.npy'. Columns: [q1..q7, x, y, z, d1..d{N_MESHES}]"
    )
