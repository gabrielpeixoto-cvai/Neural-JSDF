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

# from utils import meshes_fk # Required for Forward Kinematics


# --- ASSUMED EXTERNAL DATA AND FUNCTIONS ---
# Placeholders for required external logic. You must replace these
# with the actual functions/data from your environment.
def load_mesh_data(mat_path="meshes/mesh_light_pts.mat"):
    """
    Load mesh data. This function must return a structure 'mesh'
    similar to how it is used in gen_dataset.py.
    """
    print("WARNING: Mesh data loading is currently a placeholder.")
    # Example placeholder structure - replace with actual loading logic
    try:
        mat_contents = sio.loadmat(mat_path)
        mesh = mat_contents["mesh"][0]
        return mesh
    except:
        return None


def get_transformed_meshes(mesh, q, N_MESHES=9):
    """
    Perform Forward Kinematics (FK) to get transformed meshes.
    This function must be imported from your 'utils.py' or equivalent.
    It should return a list/array of transformed mesh structures,
    where each structure has 'V' (Vertices) and 'F' (Faces).
    """
    print("WARNING: Forward Kinematics (FK) is currently a placeholder.")
    # Example placeholder for when FK fails
    print(q.shape)
    return meshes_fk(mesh, np.eye(4), q)
    # return None


# ------------------------------------------


def visualize_sdf_points_and_meshes(
    npy_path="robot_dataset_py_10.npy", N_MESHES=9, N_PTS_PER_JPOS=990
):
    """
    Loads the dataset, selects a single joint position's data,
    and visualizes the 3D points colored by d_min alongside the
    transformed robot links for that joint position.
    """
    try:
        dataset = np.load(npy_path)
    except FileNotFoundError:
        print(f"ERROR: '{npy_path}' not found. Cannot visualize dataset.")
        return

    # 1. Extract the specific joint position and corresponding data
    # The dataset is structured as [q1..q7, x, y, z, d1..dN]
    q_all = dataset[:, 0:7]
    pts_all = dataset[:, 7:10]
    dist_arr_all = dataset[:, 10:]

    # A single joint configuration's data is the first N_PTS_PER_JPOS rows.
    # N_PTS_PER_JPOS = 9 * (25+35+20+20+10) = 990, based on gen_dataset.py

    # Selected Joint Position (q) - The first one in the dataset
    q = q_all[0, :]
    print(f"Plotting for joint position (q): {q}")

    # Subset of Points for this q
    pts_plot = pts_all[:N_PTS_PER_JPOS]
    dist_arr = dist_arr_all[:N_PTS_PER_JPOS]

    # Calculate Minimum Signed Distance (d_min) for this subset
    d_min_plot = np.min(dist_arr, axis=1)

    print(f"Plotting {pts_plot.shape[0]} query points.")

    # 2. Setup 3D Scatter Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # --- Plot the Query Points ---
    vmax = 0.1
    vmin = -0.1

    scatter = ax.scatter(
        pts_plot[:, 0],
        pts_plot[:, 1],
        pts_plot[:, 2],
        c=d_min_plot,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        s=5,
        alpha=0.6,
    )

    # --- Plot the Transformed Meshes (Robot Links) ---
    mesh = load_mesh_data()
    if mesh is not None:
        mesh_fk = get_transformed_meshes(mesh, q, N_MESHES)

        if mesh_fk is not None:
            for j in range(N_MESHES):
                V = mesh_fk[j]["V"]  # Transformed Vertices
                F = mesh_fk[j]["F"]  # Faces (0-indexed for Matplotlib)

                # Plot the mesh using plot_trisurf
                ax.plot_trisurf(
                    V[:, 0],
                    V[:, 1],
                    V[:, 2],
                    triangles=F - 1,
                    color="gray",
                    alpha=0.3,
                    edgecolor="none",
                )

    # 3. Labels, Title, and Colorbar
    ax.set_title(f"Query Points and Robot Meshes for a Single Configuration")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Adjust limits to fit both points and robot
    # You may need to manually adjust these based on your robot's size
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Minimum Signed Distance (m) to Robot Links ($d_{min}$)")

    plt.savefig("sdf_visualization_with_meshes.png")
    plt.show()  # Uncomment to run locally


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
    npy_path="robot_dataset_py.npy", N_MESHES=9, N_PTS_PER_JPOS=990
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
    pts_all = dataset[:, 7:10]
    dist_arr_all = 100 * dataset[:, 10:]
    q = dataset[0, 0:7]  # First joint position

    pts_plot = pts_all[:N_PTS_PER_JPOS]
    dist_arr = dist_arr_all[:N_PTS_PER_JPOS]
    d_min_plot = np.min(dist_arr, axis=1)

    print(f"Plotting for joint position (q): {q}")
    print(f"Plotting {pts_plot.shape[0]} query points.")
    print(f"Plotting {dist_arr.shape} query points.")

    # 2. Color the Query Points by d_min
    # Use a range centered around 0 (the surface)
    vmax = np.max(d_min_plot)
    vmin = np.min(d_min_plot)
    point_colors = map_scalar_to_color_custom(d_min_plot, vmin=vmin, vmax=vmax)

    # Create Trimesh Point Cloud for SDF data
    sdf_point_cloud = trimesh.points.PointCloud(pts_plot, colors=point_colors)

    # 3. Load and Transform Meshes
    scene_entities = [sdf_point_cloud]
    mesh = load_mesh_data()

    if mesh is not None:
        mesh_fk = get_transformed_meshes(mesh, q, N_MESHES)

        if mesh_fk is not None:
            for j in range(N_MESHES):
                if "V" in mesh_fk[j] and "F" in mesh_fk[j]:
                    V = mesh_fk[j]["V"]
                    # CRITICAL FIX: Ensure faces are 0-indexed for Trimesh
                    F = mesh_fk[j]["F"] - 1

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


diagnose_dmin_range()
# Execute the visualization function
# visualize_sdf_points_and_meshes(N_MESHES=9)
visualize_sdf_points_trimesh(N_MESHES=9)
