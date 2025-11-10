import numpy as np
import trimesh

# --- DH Transformation Matrix and FK Implementation ---


def dh_am_matrix(r, d, alpha, q):
    """
    Computes the Modified Denavit-Hartenberg (DH) transformation matrix (Am)
    used in the Franka DH convention from franka_dh_fk.m.
    """
    cq, sq = np.cos(q), np.sin(q)
    ca, sa = np.cos(alpha), np.sin(alpha)

    T = np.array(
        [
            [cq, -sq, 0, r],
            [sq * ca, cq * ca, -sa, -d * sa],
            [sq * sa, cq * sa, ca, d * ca],
            [0, 0, 0, 1],
        ]
    )
    return T


#


def franka_dh_fk(joint_state, base):
    """
    MATLAB: franka_dh_fk.m
    Computes the Forward Kinematics for the Franka Panda.

    Returns: A list of 9 4x4 homogeneous transformation matrices:
             P[0] (Base/Link0 frame) to P[8] (Hand/EE frame).
    """

    # DH parameters from meshes_fk.m / franka_dh_fk.m (8 sets of parameters)
    r = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
    d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
    alpha = np.array(
        [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
    )

    # P is a list of 9 transforms (P{1} to P{9} in MATLAB)
    P = [None] * 9

    # P[0] (MATLAB P{1}) is the base transform (Link 0 frame)
    P[0] = base

    # Kinematic chain for 7 joints (i=1 to 7)
    T_prev = base
    for i in range(1, 8):
        # DH params/joint state for link i are at index i-1 in the arrays
        T_link = dh_am_matrix(r[i - 1], d[i - 1], alpha[i - 1], joint_state[i - 1])
        T_prev = T_prev @ T_link
        P[i] = T_prev

    # Transformation for hand end-effector (P[8], MATLAB P{9})
    # Uses the last set of DH parameters (index 7) and fixed angle -pi/4
    T_hand_fixed = dh_am_matrix(r[7], d[7], alpha[7], -np.pi / 4)
    T_hand = T_prev @ T_hand_fixed
    P[8] = T_hand

    return P


def meshes_fk(mesh_data, base, joint_state):
    """
    MATLAB: meshes_fk.m
    Computes the forward kinematics and transforms the robot link meshes.

    mesh_data: A list of dicts/objects where mesh_data[i] has 'v' (vertices) and 'f' (faces).
    """

    # 1. Compute all link transforms P
    P = franka_dh_fk(joint_state, base)

    mesh_fk = []

    for i in range(len(P)):
        # P[i] is the transform from the base frame to the i-th link frame
        T_link = P[i]
        R_link = T_link[:3, :3]
        t_link = T_link[:3, 3]
        # print(f"Rlink: {R_link}")
        # print(f"tlink: {t_link}")
        # print(f"mesh data v {mesh_data[i]['v']} ")
        # print(f"P: {len(P)} meshdata: {len(mesh_data)}")

        # Transform vertices: V' = V * R_link^T + t_link^T
        # (Equivalent to MATLAB: V' = mesh{i}.v * R' + T')
        transformed_vertices = mesh_data[i]["v"][0][0] @ R_link.T + t_link.T

        mesh_fk.append(
            {
                "V": transformed_vertices,  # Transformed vertices
                "F": mesh_data[i]["f"][0][0],  # Faces remain the same
            }
        )

    # --- 2. Transform Gripper Fingers (Meshes 9 and 10) ---
    # The two finger meshes are mesh_data[9] and mesh_data[10].
    # They are transformed relative to the Hand frame (P[8]).
    P_hand = P[8]
    R_hand = P_hand[:3, :3]
    t_hand = P_hand[:3, 3]

    # Gripper Joint State for Finger Movement (from the MATLAB plots, this is j_state[7])
    # The finger motion is controlled by the 8th joint state (index 7).
    # MATLAB: T(1:3,4) = [0 joint_state(8) 0.065]'
    # In Python: j_state[7] is the 8th element (index 7)
    if joint_state.shape[0] < 8:
        finger_joint = 0
    else:
        finger_joint = joint_state[7]

    # 2a. Finger 1 (mesh_data[9], index i=9)
    # T_f1 is the offset transform for the first finger relative to the Hand.
    T_f1_offset = np.eye(4)
    # The offset is typically based on the DH parameters and the 8th joint position
    # The MATLAB plotting functions use: [0 joint_state(8) 0.065]'
    T_f1_offset[:3, 3] = [0, finger_joint, 0.065]
    T_f1 = P_hand @ T_f1_offset

    R_f1 = T_f1[:3, :3]
    t_f1 = T_f1[:3, 3]

    i = 9  # Index for the mesh_data list
    V_matrix = mesh_data[i]["v"][0][0]
    F_matrix = mesh_data[i]["f"][0][0]
    transformed_vertices = V_matrix @ R_f1.T + t_f1

    mesh_fk.append({"V": transformed_vertices, "F": F_matrix})

    # 2b. Finger 2 (mesh_data[10], index i=10)
    # T_f2 is the offset transform for the second finger relative to the Hand.
    # The second finger moves symmetrically (negative y offset)
    T_f2_offset = np.eye(4)
    T_f2_offset[:3, 3] = [0, -finger_joint, 0.065]
    T_f2 = P_hand @ T_f2_offset

    R_f2 = T_f2[:3, :3]
    t_f2 = T_f2[:3, 3]

    i = 10  # Index for the mesh_data list
    V_matrix = mesh_data[i]["v"][0][0]
    F_matrix = mesh_data[i]["f"][0][0]
    transformed_vertices = V_matrix @ R_f2.T + t_f2

    mesh_fk.append({"V": transformed_vertices, "F": F_matrix})

    return mesh_fk


#

# --- Bounding Box Helper Functions ---


def get_bbox(V):
    """
    MATLAB: getBBox.m
    Computes the axis-aligned bounding box for a set of vertices V.
    """
    if V.size == 0:
        return {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}

    V = np.asarray(V)
    bbox = {
        "xmin": np.min(V[:, 0]),
        "xmax": np.max(V[:, 0]),
        "ymin": np.min(V[:, 1]),
        "ymax": np.max(V[:, 1]),
        "zmin": np.min(V[:, 2]),
        "zmax": np.max(V[:, 2]),
    }
    return bbox


#


def scale_bbox(bbox, d):
    """
    MATLAB: scaleBBox.m
    Scales an existing bounding box by distance d in all directions.
    """
    bbox_scaled = {
        "xmin": bbox["xmin"] - d,
        "xmax": bbox["xmax"] + d,
        "ymin": bbox["ymin"] - d,
        "ymax": bbox["ymax"] + d,
        "zmin": bbox["zmin"] - d,
        "zmax": bbox["zmax"] + d,
    }
    return bbox_scaled


#


def sample_bbox(bbox, n_pts):
    """
    MATLAB: sampleBbox.m
    Samples n_pts uniformly inside the given bounding box.
    """
    points = np.random.rand(n_pts, 3)
    points[:, 0] = bbox["xmin"] + (bbox["xmax"] - bbox["xmin"]) * points[:, 0]
    points[:, 1] = bbox["ymin"] + (bbox["ymax"] - bbox["ymin"]) * points[:, 1]
    points[:, 2] = bbox["zmin"] + (bbox["zmax"] - bbox["zmin"]) * points[:, 2]
    return points


#


def point_to_mesh_signed_distance(faces, vertices, query_points):
    """
    Replaces point2trimesh and inpolyhedron functionality using Trimesh.

    Returns:
    - signed_distances: (N_pts x 1) array of signed distances.
                        Negative for inside, positive for outside.
    """
    # Faces are 1-indexed in MATLAB, convert to 0-indexed for Trimesh
    faces_0_indexed = faces - 1

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_0_indexed)

    # Calculate signed distance: Negative for inside, positive for outside
    signed_distances = trimesh.proximity.signed_distance(mesh, query_points)

    return signed_distances, mesh
