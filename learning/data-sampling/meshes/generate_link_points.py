import numpy as np
import scipy.io as sio
import trimesh
import os

# --- Geometry Helper Function (already in utils.py, but included for completeness) ---


def get_bbox(V):
    """Computes the axis-aligned bounding box."""
    if V.size == 0:
        return {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
    V = np.asarray(V)
    return {
        "xmin": np.min(V[:, 0]),
        "xmax": np.max(V[:, 0]),
        "ymin": np.min(V[:, 1]),
        "ymax": np.max(V[:, 1]),
        "zmin": np.min(V[:, 2]),
        "zmax": np.max(V[:, 2]),
    }


def scale_bbox(bbox, d):
    """Scales an existing bounding box by distance d in all directions."""
    return {
        "xmin": bbox["xmin"] - d,
        "xmax": bbox["xmax"] + d,
        "ymin": bbox["ymin"] - d,
        "ymax": bbox["ymax"] + d,
        "zmin": bbox["zmin"] - d,
        "zmax": bbox["zmax"] + d,
    }


def sample_bbox(bbox, n_pts):
    """Samples n_pts uniformly inside the given bounding box."""
    points = np.random.rand(n_pts, 3)
    points[:, 0] = bbox["xmin"] + (bbox["xmax"] - bbox["xmin"]) * points[:, 0]
    points[:, 1] = bbox["ymin"] + (bbox["ymax"] - bbox["ymin"]) * points[:, 1]
    points[:, 2] = bbox["zmin"] + (bbox["zmax"] - bbox["zmin"]) * points[:, 2]
    return points


def point_to_mesh_signed_distance(faces, vertices, query_points):
    """
    Replaces point2trimesh and inpolyhedron functionality using Trimesh.
    Returns signed distances (negative for inside, positive for outside).
    """
    # Faces are 1-indexed in MATLAB, convert to 0-indexed for Trimesh
    faces_0_indexed = faces[0][0] - 1

    # Vertices must be unwrapped, similar to the fix in meshes_fk
    vertices_unwrapped = vertices[0][0]

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices_unwrapped, faces=faces_0_indexed)

    mesh.fix_normals()

    # --- ROBUSTNESS CHECK ---
    # Signed distance is only truly accurate for watertight meshes.
    # We can still run it, but it's good to know if the mesh has holes.
    if not mesh.is_watertight:
        print(
            f"  WARNING: Mesh is not watertight. Signed distance results may be inaccurate."
        )
    # ----------------------

    # Calculate signed distance
    signed_distances = trimesh.proximity.signed_distance(mesh, query_points)

    return signed_distances, mesh


# --- Main Point Generation Script ---


def generate_link_points(
    input_mat_path="mesh_light.mat",
    output_mat_path="mesh_light_pts.npz",
):
    """
    Converts MATLAB linkPtsGen.m to Python. Generates interior and close-to-surface
    points for each link mesh and saves the enhanced mesh structure.
    """

    # Setup parameters from MATLAB file
    N_MESHES = 11
    N_SAMPLE_PTS = 10000
    BBOX_SCALE_FACTOR = 0.02
    CLOSE_DISTANCE_THRESHOLD = 0.03

    print(f"Loading meshes from: {input_mat_path}")

    # 1. Load the initial mesh structure
    try:
        mat_contents = sio.loadmat(input_mat_path)
        # Assuming 'mesh' is the key holding the cell array structure
        mesh_data = mat_contents["mesh"][0]
    except FileNotFoundError:
        print(
            f"ERROR: Could not find '{input_mat_path}'. Please check the path and ensure the file exists."
        )
        return

    # Dictionary to store all arrays for saving
    data_to_save = {}

    # 2. Iterate through all 11 meshes and generate points
    for j in range(N_MESHES):
        print(f"Processing mesh {j+1}/{N_MESHES}...")

        # Get Vertices and Faces (assuming the nested structure [0][0] confirmed earlier)
        V = mesh_data[j]["v"]
        F = mesh_data[j]["f"]

        # --- Point Generation and Signed Distance Calculation ---

        # 1. Generate points inside an expanded BBox
        # MATLAB: bbox_inside = getBBox(mesh{j}.v);
        # MATLAB: bbox_inside = scaleBBox(bbox_inside, 0.2);

        # The bounding box is calculated on the local mesh vertices
        bbox_local = get_bbox(V[0][0])
        bbox_sample = scale_bbox(bbox_local, BBOX_SCALE_FACTOR)

        pts_all = sample_bbox(bbox_sample, N_SAMPLE_PTS)

        # 2. Calculate signed distance for all points
        # MATLAB: [dst, pt_closest] = point2trimesh(...)
        # MATLAB: IN = inpolyhedron(...)
        # MATLAB: dst2(IN) = -1 * dst2(IN);

        # In Python, this is replaced by one Trimesh call:
        # Negative for inside, positive for outside
        signed_distances, trimesh_mesh = point_to_mesh_signed_distance(F, V, pts_all)
        points_inside = trimesh_mesh.contains(pts_all)
        # print(points_inside)
        # 3. Filter and store 'int_pts' and 'close_pts'

        # Interior points: dst2 < 0
        signed_distances_abs = abs(signed_distances)
        # print(signed_distances.shape)
        # print(signed_distances)
        # print(signed_distances_abs.shape)
        # print(signed_distances_abs)
        # print(pts_all.shape)
        signed_distances_abs[np.where(points_inside)] = (
            -1 * signed_distances_abs[np.where(points_inside)]
        )
        # print(signed_distances_abs[np.where(points_inside)])
        lbl_inside = np.where(signed_distances_abs < 0)[0]
        # print(lbl_inside)
        # print(signed_distances)
        int_pts = pts_all[lbl_inside, :]

        # Close points: dst2 > 0 AND dst2 < 0.03
        # The original MATLAB snippet used dst2>0 & dst2<0.03
        lbl_close = np.where(
            (signed_distances_abs > 0)
            & (signed_distances_abs < CLOSE_DISTANCE_THRESHOLD)
        )[0]
        # print(lbl_close)
        close_pts = pts_all[lbl_close, :]

        # 4. Store all data for this mesh in the output dictionary
        # We use a prefix for each mesh (e.g., "mesh_0_", "mesh_1_")
        prefix = f"mesh_{j}_"

        # Store new points
        data_to_save[prefix + "int_pts"] = int_pts
        data_to_save[prefix + "close_pts"] = close_pts

        # Store original fields (unwrapping them from MATLAB's structure)
        field_names = mesh_data[0].dtype.names
        for name in field_names:
            original_data = mesh_data[j][name]
            unwrapped_data = original_data

            # Proactively unwrap data if it matches the V/F (1,1) object array structure
            if (
                isinstance(original_data, np.ndarray)
                and original_data.dtype == "object"
                and original_data.shape == (1, 1)
            ):
                unwrapped_data = original_data[0][0]

            data_to_save[prefix + name] = unwrapped_data

        print(
            f"  -> Generated {int_pts.shape[0]} interior and {close_pts.shape[0]} close points."
        )

    # 3. Save the enhanced mesh data structure
    print(f"\nSaving enhanced mesh data to: {output_mat_path}")

    # Use np.savez_compressed to save the dictionary as a .npz file
    # The **data_to_save unpacks the dictionary into keyword arguments
    try:
        np.savez_compressed(output_mat_path, **data_to_save)
        print("Point generation and saving complete.")
    except Exception as e:
        print(f"ERROR: Failed to save .npz file: {e}")

    print("Point generation and saving complete.")


if __name__ == "__main__":
    # You may need to adjust the path depending on where your 'meshes' folder is located
    # Example: os.path.join(os.path.dirname(__file__), '../meshes/mesh_light.mat')
    generate_link_points()
