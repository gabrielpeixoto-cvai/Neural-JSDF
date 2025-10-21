import numpy as np
import scipy.io as sio  # For loading .mat files
from utils import (
    get_bbox,
    scale_bbox,
    sample_bbox,
    meshes_fk,
    point_to_mesh_signed_distance,
)

# --- Main Dataset Generation Script ---

if __name__ == "__main__":

    # 1. Setup and Initialization

    # Load mesh data - Adjust path if necessary.
    try:
        # Assuming the .mat file contains a structure/cell array 'mesh'
        # MATLAB cell array 'mesh' is typically loaded as a structured NumPy array in Python
        mat_contents = sio.loadmat("meshes/mesh_light_pts.mat")
        mesh = mat_contents["mesh"][0]
    except FileNotFoundError:
        print(
            "ERROR: Could not find 'meshes/mesh_light_pts.mat'. Please check file path."
        )
        exit()

    base = np.eye(4)

    # Joint limits (from genDataset.m)
    q_min_raw = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0]
    )
    q_max_raw = np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04]
    )

    # Inflate limits by 5% as per MATLAB code
    q_span = q_max_raw - q_min_raw
    q_min = q_min_raw - q_span * 0.05
    q_max = q_max_raw + q_span * 0.05
    #

    # 2. Dataset Configuration
    N_MESHES = len(mesh) - 2  # Assuming only the first N-2 meshes are for collision
    N_JPOS = 10  # Number of joint positions to sample
    # N_JPOS = 5000  # Uncomment for the value used in the paper

    # Points per mesh per type (from genDataset.m)
    N_INSIDE = np.full(N_MESHES, 25)
    N_OUTSIDE = np.full(N_MESHES, 35)
    N_CLOSE = np.full(N_MESHES, 20)
    N_FAR = np.full(N_MESHES, 20)
    N_ZERO = np.full(N_MESHES, 10)

    box_delta = 0.1
    # Bounding box for 'far' points (global workspace)
    bbox_far = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1, "zmin": -1, "zmax": 1}
    #

    all_data = []

    # 3. Main Data Generation Loop
    for i in range(N_JPOS):
        print(f"Generating data for joint position {i+1}/{N_JPOS}...")

        # Sample a random joint position (q)
        q_rand = q_min + np.random.rand(1, len(q_min)) * (q_max - q_min)
        jpos = q_rand[0]  # Joint position vector

        # Compute FK and transform meshes
        mesh_fk = meshes_fk(mesh, base, jpos)

        pts_all = np.empty((0, 3))

        # Iterate through each relevant robot link mesh
        for j in range(N_MESHES):
            V = mesh_fk[j]["V"]  # Transformed Vertices
            F = mesh_fk[j]["F"]  # Faces

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
            V = mesh_fk[j]["V"]
            F = mesh_fk[j]["F"]

            # point_to_mesh_signed_distance handles the 1-indexing conversion internally
            signed_distances = point_to_mesh_signed_distance(F, V, pts_all)

            dist_arr[:, j] = signed_distances
        #

        # 5. Create and Store Dataset Entry
        # The joint state (jpos) is 8 elements. We only use the first 7 for the dataset.
        joint_cols = np.tile(jpos[:7], (n_pts, 1))

        # subdataset = [q(1:7), pts_all (x,y,z), dist_arr (d1...dN)]
        subdataset = np.hstack([joint_cols, pts_all, dist_arr])
        all_data.append(subdataset)

    # Final Concatenation and Save
    final_dataset = np.vstack(all_data)

    # Save the dataset to a file (e.g., NumPy .npy or CSV)
    np.save("robot_dataset.npy", final_dataset)
    # np.savetxt('robot_dataset.csv', final_dataset, delimiter=',')

    print(f"\nDataset generation complete. Total samples: {final_dataset.shape[0]}")
    print(
        f"Data saved to 'robot_dataset.npy'. Columns: [q1..q7, x, y, z, d1..d{N_MESHES}]"
    )
