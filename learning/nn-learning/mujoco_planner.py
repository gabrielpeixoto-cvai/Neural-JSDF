import mujoco as mj
from mujoco import MjSpec, MjModel, MjData, mj_name2id, mj_jacSite
import mujoco.viewer  # <--- NEW IMPORT
import numpy as np
import time
import traceback
from typing import Optional, List, Tuple

# Import OMPL libraries
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("WARNING: OMPL Python bindings not found. MotionPlannerOMPL will not run.")
    ou, ob, og = None, None, None

import fcl


class RobotPlanner:
    """
    A simple MuJoCo class for robot planning with model loading,
    state access, Inverse Kinematics (IK), and collision detection.
    """

    NUM_VIZ_MARKERS = 1000
    CAMERA_NAME = "eef_camera"
    MARKER_PREFIX = "viz_marker_"

    def __init__(self, mjcf_path: str, end_effector_site_name: str):
        """
        1. Loads the robot model using mjspec from an MJCF file.
           The mjspec object is kept for potential future programmatic additions.
        """
        # Load MJCF into MjSpec for programmatic editing
        self.spec = MjSpec.from_file(mjcf_path)
        self.end_effector_site_name = end_effector_site_name

        self._add_viz_geoms_to_spec()
        self._add_camera_to_spec()
        self._add_obstacles_to_spec()

        # Compile the spec into mjModel and create MjData
        self.model = self.spec.compile()
        self.data = MjData(self.model)
        # Access mass and inertia
        for body_id in range(self.model.nbody):
            body_name = mj.mj_id2name(
                self.model, mj.mjtObj.mjOBJ_BODY, body_id
            )  # Get body name (optional)
            mass = self.model.body_mass[body_id]
            inertia = self.model.body_inertia[body_id]
            group = self.model.body
            print(
                f"Body {body_id} (Name: {body_name}): Mass = {mass}, Inertia = {inertia}"
            )

        # Get the ID for the end-effector site
        self.eef_site_id = mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, self.end_effector_site_name
        )

        if self.eef_site_id == -1:
            raise ValueError(f"Site '{end_effector_site_name}' not found in model.")

        # --- Sensing and Visualization IDs ---
        self.camera_id = mj_name2id(
            self.model, mj.mjtObj.mjOBJ_CAMERA, self.CAMERA_NAME
        )

        print(f"CAMERAID : {self.camera_id}")
        self.marker_body_ids, self.marker_geom_ids = self._get_marker_ids()

        # --- Joint Identification ---
        # For geometric planning, we plan over all DOFs (model.nq)
        self.num_dofs = self.model.nq

        print(f"RobotPlanner initialized. Planning DOFs (nq): {self.num_dofs}")
        # Call forward once to initialize all data structures
        mj.mj_forward(self.model, self.data)

        # Off-screen rendering context variables
        self.renderer = None
        self.offscreen_context = None
        print(self.marker_body_ids)

        # Initialize Mocap Quaternions for visualization bodies
        # MOCAP geoms are placed inside a MOCAP body, the body's orientation is controlled here.
        for body_id in self.marker_body_ids:
            mocap_slot = self.model.body_mocapid[body_id]
            if mocap_slot != -1:
                # Set identity quaternion for all visualization markers
                self.data.mocap_quat[mocap_slot] = [1.0, 0.0, 0.0, 0.0]

        # New FCL initialization
        self.fcl_objects = []
        self._init_fcl()

    def _add_viz_geoms_to_spec(self):
        """Programmatically adds MOCAP bodies and their geoms to the worldbody."""
        for i in range(self.NUM_VIZ_MARKERS):
            body_name = f"{self.MARKER_PREFIX}{i}"

            # Create a mocap body in the worldbody
            mocap_body = self.spec.worldbody.add_body(
                name=body_name,
                pos=[-1, -1, -1],
                mocap=True,  # Critical for dynamic positioning via data.mocap_pos
            )

            # Add a visualization geom to the mocap body
            mocap_body.add_geom(
                name=f"geom_{body_name}",  # Geom must have a unique name
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],  # Small sphere size
                pos=[0, 0, 0],
                rgba=[0, 1, 0, 1.0],  # Start invisible (alpha 0.0)
                group=3,
            )
        print(
            f"Added {self.NUM_VIZ_MARKERS} visualization MOCAP bodies programmatically."
        )

    def _add_obstacles_to_spec(self):
        """Programmatically adds MOCAP bodies and their geoms to the worldbody."""
        positions = [[-1, 0, 0.5], [1, 0, 0.5], [0, -1, 0.5], [0, 1, 0.5]]
        sizes = [[0.02, 1, 0.5], [0.02, 1, 0.5], [1, 0.02, 0.5], [1, 0.02, 0.5]]
        for i in range(4):
            body_name = f"obstacle_{i}"

            # Create a mocap body in the worldbody
            obstacle_body = self.spec.worldbody.add_body(
                name=body_name,
                pos=positions[i],
                # mocap=True,  # Critical for dynamic positioning via data.mocap_pos
            )

            # Add a visualization geom to the mocap body
            obstacle_body.add_geom(
                name=f"geom_{body_name}",  # Geom must have a unique name
                type=mj.mjtGeom.mjGEOM_BOX,
                size=sizes[i],  # Small sphere size
                pos=[0, 0, 0],
                rgba=[0, 1, 0, 1.0],  # Start invisible (alpha 0.0)
                group=2,
            )
        print(f"Added {4} obstacle bodies programmatically.")

    def _add_camera_to_spec(self):
        """Programmatically adds the camera to the end-effector body."""
        # Find the end-effector body in the spec
        # print(dir(mj))
        # print(dir(self.spec.worldbody))
        eef_body_spec = self.spec.worldbody.find_child(self.end_effector_site_name)
        print(f"EEBODY: {eef_body_spec.name}")

        if eef_body_spec:
            eef_body_spec.add_camera(
                name=self.CAMERA_NAME,
                pos=[0, 0, 0.05],
                # Default MuJoCo camera alignment
                xyaxes=[1, 0, 0, 0, 0, -1],
                fovy=60,
                # mode=mj.mjtCamera,  # Camera pose is fixed relative to the body
            )
            print(
                f"Added camera '{self.CAMERA_NAME}' to body '{self.end_effector_site_name}' programmatically."
            )
        else:
            raise ValueError(
                f"Could not find body '{self.end_effector_site_name}' in spec to attach camera."
            )

    def _get_marker_ids(self) -> Tuple[List[int], List[int]]:
        """Collects IDs for all pre-allocated visualization geoms."""
        body_ids = []
        geom_ids = []
        for i in range(self.NUM_VIZ_MARKERS):
            body_id = mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, f"{self.MARKER_PREFIX}{i}"
            )
            geom_id = mj_name2id(
                self.model, mj.mjtObj.mjOBJ_GEOM, f"geom_{self.MARKER_PREFIX}{i}"
            )
            if body_id != -1:
                body_ids.append(body_id)
            if geom_id != -1:
                geom_ids.append(geom_id)

        return body_ids, geom_ids

    def setup_offscreen_rendering(self, width: int = 640, height: int = 480):
        """Initializes the off-screen rendering context for sensing."""
        if not self.renderer:
            self.renderer = mj.Renderer(self.model, height=height, width=width)
            # print(dir(mj))
            # print(dir(self.renderer.scene))
            # self.renderer.set_camera(self.camera_id)
            # self.renderer.update_scene()
            print(f"Off-screen renderer set up for camera '{self.CAMERA_NAME}'.")

    def release_offscreen_rendering(self):
        """Releases the off-screen rendering context."""
        self.renderer = None

    def step_simulation(self):
        """Advances the simulation by one step."""
        mj.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        """
        2. Gets the current joint positions (generalized coordinates).
        """
        # data.qpos contains the generalized positions (joint positions)
        # Note: Depending on your model, this may include a 7-element free-body
        # position/orientation for the world body, followed by revolute/prismatic joints.
        return self.data.qpos.copy()

    def set_joint_positions(self, q: np.ndarray):
        """Sets the positions of all joints in data.qpos."""
        if q.shape[0] != self.model.nq:
            raise ValueError(
                f"Input q size ({q.shape[0]}) must match model.nq ({self.model.nq})."
            )
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)

    def get_tcp_position(self) -> np.ndarray:
        """
        2. Gets the TCP (Tool Center Point) position from the specified site.
        """
        # data.site_xpos contains the 3D position of all sites
        return self.data.xpos[self.eef_site_id].copy()

    def get_tcp_orientation(self) -> np.ndarray:
        """
        Gets the TCP (Tool Center Point) orientation (rotation matrix) from the site.
        """
        # data.site_xmat contains the 3x3 rotation matrix for all sites
        # It's stored as a flattened 9-element array
        return self.data.xmat[self.eef_site_id].reshape((3, 3)).copy()

    def check_joint_limits(self, q: np.ndarray) -> bool:
        """
        Checks if the given full qpos vector is within model limits.
        We iterate through all joints and check their respective ranges.
        """
        if q.shape[0] != self.model.nq:
            # Should be caught by OMPL state space checks, but useful here.
            return False

        # The joint range is stored in model.jnt_range[joint_id]
        # model.jnt_qposadr provides the index into qpos for the start of each joint
        # For simple hinge/slide joints, this is a 1-to-1 map.

        for jnt_id in range(self.model.njnt):
            if self.model.jnt_limited[jnt_id]:
                qpos_index = self.model.jnt_qposadr[jnt_id]

                # Assume simple 1-DOF joints (hinge/slide) for now.
                # Complex joints (e.g., freejoint, ball) need specialized handling.
                if self.model.jnt_type[jnt_id] in [
                    mj.mjtJoint.mjJNT_HINGE,
                    mj.mjtJoint.mjJNT_SLIDE,
                ]:
                    limit_min = self.model.jnt_range[jnt_id, 0]
                    limit_max = self.model.jnt_range[jnt_id, 1]

                    if not (limit_min <= q[qpos_index] <= limit_max):
                        return False

        return True

    def execute_ik_jacobian(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_steps: int = 100,
        tolerance: float = 1e-3,
        damping: float = 1e-1,
    ) -> np.ndarray:
        """
        3. Executes a simple Inverse Kinematics using the damped pseudo-inverse
           of the full (position and orientation) Jacobian.

        Note: This is an iterative *control* method, not an offline *solver*.
              It modifies data.qpos in place over multiple steps.
        """

        qpos_current = self.data.qpos.copy()

        # Determine the degrees of freedom (number of controllable joints)
        nv = self.model.nv

        # Initialize Jacobian matrices for position and rotation (3xnv)
        jac_p = np.zeros((3, nv))
        jac_r = np.zeros((3, nv))

        # Pre-allocate the rotation matrix result for mju_quat2Mat (shape 9)
        # FIX: Define the target matrix array here
        target_mat = np.zeros(9)

        for step in range(max_steps):
            # 1. Update kinematics based on current qpos
            mj.mj_forward(self.model, self.data)

            # 2. Get current EEF pose
            eef_pos = self.get_tcp_position()
            eef_mat = self.get_tcp_orientation()

            # 3. Compute pose error
            pos_error = target_pos - eef_pos

            # Orientation error calculation using the formula:
            # e_rot = 0.5 * (R_eef @ R_target_transposed - R_target @ R_eef_transposed) VEC
            # A simpler approach using delta rotation (a.k.a. orientation error vector) is:
            # target_mat = mj.mju_quat2Mat(target_quat)
            # Rotation error vector (3-element): R_eef * (R_eef_T * R_target)_skew_inv
            # MuJoCo has a helper for the error:
            rot_error = np.zeros(3)
            # mj.mju_quatDiff(self.data.xquat[self.eef_site_id], target_quat, rot_error)
            mj.mju_subQuat(
                rot_error,
                target_quat,  # qa (Target quaternion)
                self.data.xquat[self.eef_site_id],  # qb (Current quaternion)
            )

            # Combined pose error (6-element)
            pose_error = np.hstack([pos_error, rot_error])

            # Check for convergence
            if np.linalg.norm(pose_error) < tolerance:
                # print(f"IK converged in {step} steps.")
                return self.data.qpos.copy()

            # 4. Compute Jacobian at the site
            # jac_p and jac_r are overwritten
            mj.mj_jacBody(self.model, self.data, jac_p, jac_r, self.eef_site_id)

            # Full Jacobian (6xnv)
            J = np.vstack([jac_p, jac_r])

            # 5. Compute the Damped Pseudo-Inverse: J_pinv = J_T * (J * J_T + lambda * I)^-1
            # (A less computationally expensive way than the full pseudo-inverse for control)
            I = np.eye(6) * damping**2
            J_T = J.T
            J_pinv = J_T @ np.linalg.inv(J @ J_T + I)

            # 6. Compute velocity update
            # We treat the error as a desired Cartesian velocity
            dq = J_pinv @ pose_error

            # 7. Update joint positions (integrate velocity)
            # Use a small learning rate/step size
            qpos_current += 0.5 * dq

            # Update MuJoCo state for the next step (important for mj_forward/mj_jacSite)
            self.data.qpos[:] = qpos_current

        # print("IK reached max steps without converging.")
        return self.data.qpos.copy()  # Return best effort

    """
    def detect_collision(self) -> bool:
        #4. Detects if there are any active collisions using MuJoCo contacts.
        # mj_step (or mj_forward which is called inside mj_step) populates data.ncon
        # Check if the number of active contacts is greater than 0
        # print(self.data.ncon)
        return self.data.ncon > 0
    """

    def detect_collision(self, method="mujoco") -> bool:
        """Generic entry point for collision checking."""
        mj.mj_forward(self.model, self.data)
        if method == "mujoco":
            # MuJoCo needs forward kinematics to update contacts
            return self.data.ncon > 0
        elif method == "fcl":
            # print("Here")
            # FCL only needs the forward kinematics for the transforms
            return self.detect_collision_fcl()
        elif method == "fcl-fast":
            return self.detect_collision_fcl_fast()
        return False

    def get_contacts_info(self):
        """
        Utility to get detailed collision information.
        """
        if not self.detect_collision():
            return "No active contacts detected."

        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get geom names for better readability
            geom1_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom2_id)

            contacts.append(
                {
                    "geom1": geom1_name,
                    "geom2": geom2_name,
                    "position": contact.pos.copy(),
                    "force": contact.efc_force.copy(),  # The force applied to resolve the constraint/contact
                }
            )

        return contacts

    def simulate_and_render(self, duration: float = 5.0):
        """
        Renders the current model using mujoco.viewer for a specified duration.
        Since physics are not required, time is manually advanced using the
        model's timestep to control the loop duration.
        """

        # Get the fixed simulation timestep
        timestep = self.model.opt.timestep

        # Get the current time for loop control
        current_time = 0.0

        # NOTE: We must call mj_forward before the viewer loop
        # to ensure the initial state is computed for rendering.
        mj.mj_forward(self.model, self.data)

        print(f"\n--- Launching Viewer: Displaying state for {duration} seconds ---")

        with mj.viewer.launch_passive(self.model, self.data) as viewer:

            # Loop as long as the viewer is active and the duration hasn't elapsed
            while viewer.is_running() and current_time < duration:

                # --- State Update (Manual Time Advance) ---

                # 1. Manually update the data.time (required by some visualization features)
                # We use the model's timestep to ensure consistency.
                self.data.time += timestep
                current_time += timestep

                # 2. Update the viewer
                viewer.sync()

                # 3. Sleep to match the real-time speed of the simulation
                # This makes the 5-second duration in the loop take approx 5 real-world seconds.
                time.sleep(timestep)

            print("--- Viewer closed or display duration elapsed ---")

    def disable_markers(self):
        """Sets all visualization geoms to disabled (visible=0)."""
        for geom_id in self.marker_geom_ids:
            # Group 3 is often used for visualization geoms
            self.model.geom_group[geom_id] = 3
            # Disable visualization (visible=0 means do not render)
            self.model.geom_rgba[geom_id, 3] = 0.0

    def visualize_geoms(
        self, points_xyz: np.ndarray, rgba: np.ndarray = np.array([0, 1, 0, 1])
    ):
        """
        Moves the pre-allocated visualization geoms to match the given points.
        The number of points is capped by NUM_VIZ_MARKERS.
        """

        # Ensure markers are disabled first
        self.disable_markers()

        n_points = min(len(points_xyz), self.NUM_VIZ_MARKERS)

        print(
            f"Visualizing {n_points} points using {len(self.marker_body_ids)} markers."
        )

        for i in range(n_points):
            geom_id = self.marker_geom_ids[i]
            body_id = self.marker_body_ids[i]
            mocap_id = self.model.body_mocapid[body_id]

            # Set the position (xpos) of the geom in the world frame
            # MuJoCo sets geom position in the body's frame, but since these
            # geoms are defined in the worldbody, xpos is the world position.
            self.data.mocap_pos[mocap_id][:] = points_xyz[i, :]
            # print(f"point: {points_xyz[i]} mocap: {self.data.mocap_pos[mocap_id]}")

            # Enable visualization (alpha=1.0) and set color
            self.model.geom_rgba[geom_id, :] = rgba
            self.model.geom_rgba[geom_id, 3] = 1.0  # Set alpha to 1.0
            self.model.geom_group[geom_id] = 2

        # Must call mj_forward to update geometry positions for the viewer
        mj.mj_forward(self.model, self.data)

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()
        return rgb, depth

    def rgbd_to_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intr: np.ndarray,
        extr: np.ndarray,
        width: int,
        height: int,
        depth_trunc: float = 20.0,
    ):
        cc, rr = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
        valid = (depth > 0) & (depth < depth_trunc)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
        y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
        xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
        color = rgb.transpose([2, 0, 1]).reshape((3, -1)).T / 255.0
        mask = np.isnan(xyz[:, 2])
        xyz = xyz[~mask]
        color = color[~mask]
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_t = (extr @ xyz_h.T).T
        xyzrgb = np.hstack([xyz_t[:, :3], color])
        return xyzrgb

    # === Sensing Feature ===
    def _downsample_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Uniformly downsamples the point cloud to match the number of available markers.
        """
        n_points = point_cloud.shape[0]
        n_markers = self.NUM_VIZ_MARKERS

        if n_points <= n_markers:
            print(
                f"Point cloud size ({n_points}) is less than or equal to marker count ({n_markers}). No downsampling needed."
            )
            return point_cloud

        # Use np.random.choice to select n_markers indices uniformly
        # replace=False ensures we sample unique points
        selected_indices = np.random.choice(n_points, size=n_markers, replace=False)

        downsampled_cloud = point_cloud[selected_indices, :]
        print(f"Downsampled {n_points} points to {n_markers} points for visualization.")
        return downsampled_cloud

    def capture_point_cloud(self, far_clip: float = 3.0) -> Optional[np.ndarray]:
        """
        Captures a depth map from the eef_camera and converts it to a 3D point cloud
        in the world frame.
        """
        if not self.renderer:
            print(
                "Error: Off-screen renderer not set up. Call setup_offscreen_rendering()."
            )
            return None

        # 1. Ensure latest robot state is forwarded
        mj.mj_forward(self.model, self.data)

        # 2. Render the depth image
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.CAMERA_NAME)
        depth = self.renderer.render()
        # depth = self.renderer.get_depth_np()

        # Get width and height from the renderer
        width, height = self.renderer.width, self.renderer.height

        # 3. Convert depth map to point cloud
        # Use mjd_camera_helper to transform depth to (x,y,z) coordinates
        # mj.mjd_camera_helper(
        #    self.model,
        #    self.data,
        #    points,
        #    depth.flatten(),
        #    self.camera_id,
        #    0,  # camera type (0=fixed)
        #    width,
        #    height,
        #    far_clip,  # z-near/z-far are read from camera geom data
        # )
        # points = self.depth_to_world(depth, self.camera_id, width, height)
        # from: https://github.com/google-deepmind/mujoco/issues/1863#issuecomment-2292140372
        # Intrinsic matrix.
        fov = self.model.cam_fovy[self.camera_id]
        theta = np.deg2rad(fov)
        fx = width / 2 / np.tan(theta / 2)
        fy = height / 2 / np.tan(theta / 2)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Extrinsic matrix.
        cam_pos = self.data.cam_xpos[self.camera_id]
        cam_rot = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        extr = np.eye(4)
        extr[:3, :3] = cam_rot.T
        extr[:3, 3] = cam_pos
        rgb, depth = self.render_rgbd()
        points = self.rgbd_to_pointcloud(rgb, depth, intr, extr, width, height)

        # 4. Filter out points beyond the far clip (where depth is 1.0)
        # and points at origin (0,0,0) which can be invalid.
        # valid_indices = np.where(
        #    (depth.flatten() < 1.0) & (np.linalg.norm(points, axis=1) > 1e-4)
        # )[0]

        print(f"depth shape: {depth.shape}")
        print(f"Captured {len(points)} valid points from camera.")

        return self._downsample_point_cloud(points[:, :3])

    def play_trajectory(
        self,
        trajectory: np.ndarray,
        speed_factor: float = 1.0,
        duration: Optional[float] = None,
    ):
        """
        Plays a pre-calculated joint trajectory using the MuJoCo viewer.

        Args:
            trajectory (np.ndarray): N x DOFs array of joint positions.
            speed_factor (float): Multiplier for playback speed (1.0 = real-time based on timestep).
            duration (Optional[float]): If set, overrides speed_factor to fix total duration.
        """
        if trajectory is None or trajectory.size == 0:
            print("Error: Trajectory buffer is empty.")
            return

        # Select key points to visualize (using trajectory markers)
        path_indices = np.linspace(
            0, trajectory.shape[0] - 1, self.NUM_VIZ_MARKERS, dtype=int
        )
        path_points_q = trajectory[path_indices]

        # Temporarily store the initial state to reset later
        q_initial = self.data.qpos.copy()

        # Compute the world position of the TCP for each marker location
        marker_positions = []
        for q in path_points_q:
            self.set_joint_positions(q)
            mj.mj_forward(self.model, self.data)
            marker_positions.append(self.get_tcp_position().copy())

        # Convert to numpy array and visualize the full path
        path_xyz = np.array(marker_positions)
        self.visualize_geoms(
            path_xyz, rgba=np.array([1, 0.5, 0, 1])
        )  # Orange path markers

        timestep = self.model.opt.timestep
        num_steps = trajectory.shape[0]

        # Calculate delay based on desired speed or total duration
        if duration:
            delay = duration / num_steps
        else:
            delay = timestep / speed_factor

        print(f"\n--- Playing Trajectory ({num_steps} steps, Delay: {delay:.4f}s) ---")

        # Set the robot to the first point of the trajectory
        self.set_joint_positions(trajectory[0])
        mj.mj_forward(self.model, self.data)

        with mj.viewer.launch_passive(self.model, self.data) as viewer:

            start_time = time.time()
            step_count = 0

            while viewer.is_running() and step_count < num_steps:

                q_next = trajectory[step_count]

                # Set new joint configuration
                self.set_joint_positions(q_next)

                # Forward kinematics and contact check
                mj.mj_forward(self.model, self.data)

                # Optional: Update time manually for better viewer time display
                self.data.time = time.time() - start_time

                # Update viewer and control playback speed
                viewer.sync()

                # Wait for the calculated delay
                time.sleep(delay)

                step_count += 1

            print("--- Trajectory playback finished ---")
        # Reset robot and disable visualization markers after viewer closes
        self.set_joint_positions(q_initial)
        self.disable_markers()
        mj.mj_forward(self.model, self.data)

    # FCL
    """
    def _init_fcl(self):
        #Pre-allocate FCL collision objects for all geoms in the model.
        for i in range(self.model.ngeom):
            # Skip visualization-only geoms if desired (checking geom_group)
            if self.model.geom_group[i] > 2:
                continue

            fcl_geom = self.mj_geom_to_fcl(self.model, i)
            # Create a transform (will be updated every check)
            transform = fcl.Transform()
            obj = fcl.CollisionObject(fcl_geom, transform)

            # Store with geom_id so we know which is which
            self.fcl_objects.append({"obj": obj, "geom_id": i})
    """

    def _init_fcl(self):
        """Pre-allocate FCL collision objects and initialize the manager."""
        self.fcl_objects = []
        # Two Broadphase Managers
        self.robot_manager = fcl.DynamicAABBTreeCollisionManager()
        self.env_manager = fcl.DynamicAABBTreeCollisionManager()

        # Track items in separate lists for efficient transform updates
        self.fcl_robot_items = []
        self.fcl_env_items = []

        for i in range(self.model.ngeom):
            if self.model.geom_group[i] > 2:
                continue

            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or ""
            fcl_geom = self.mj_geom_to_fcl(self.model, i)
            transform = fcl.Transform()
            obj = fcl.CollisionObject(fcl_geom, transform)

            # Store metadata in the 'user_data' if your FCL binding supports it,
            # otherwise we track it via a dictionary
            item = {"obj": obj, "geom_id": i, "name": geom_name}
            self.fcl_objects.append(item)

            # Partitioning Logic:
            # 1. Use Group 2 (assigned in _add_obstacles_to_spec and add_dynamic_obstacle)
            # 2. Backup: Check if "obstacle" is in the name
            if "obstacle" in geom_name.lower():
                self.fcl_env_items.append(item)
                self.env_manager.registerObject(obj)
                print(f" (Obstacle) Name (id): {geom_name} {i}")
            else:
                print(f" (Robot) Name (id): {geom_name} {i}")
                self.fcl_robot_items.append(item)
                self.robot_manager.registerObject(obj)

        self.robot_manager.setup()
        self.env_manager.setup()
        self._init_self_collision_pairs()
        print(
            f"FCL Partitioned: {len(self.fcl_robot_items)} Robot geoms, {len(self.fcl_env_items)} Env geoms."
        )

    def detect_collision_fcl(self) -> bool:
        """Collision detection using FCL with MuJoCo-style filtering."""
        # 1. Update transforms of all FCL objects
        for item in self.fcl_objects:
            g_id = item["geom_id"]
            pos = self.data.geom_xpos[g_id]
            mat = self.data.geom_xmat[g_id].reshape(3, 3)
            item["obj"].setTransform(fcl.Transform(mat, pos))

        # 2. Perform Pairwise Checks
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()

        for i in range(len(self.fcl_objects)):
            for j in range(i + 1, len(self.fcl_objects)):
                item1 = self.fcl_objects[i]
                item2 = self.fcl_objects[j]

                id1 = item1["geom_id"]
                id2 = item2["geom_id"]

                # --- NEW FILTERING LOGIC ---

                # A. Check Bitmasks (contype and conaffinity)
                # This is the most common way MuJoCo filters collisions
                type1, aff1 = (
                    self.model.geom_contype[id1],
                    self.model.geom_conaffinity[id1],
                )
                type2, aff2 = (
                    self.model.geom_contype[id2],
                    self.model.geom_conaffinity[id2],
                )

                """
                if not ((type1 & aff2) or (type2 & aff1)):
                    print(
                        f"Skipping {id1} vs {id2} due to bitmasks: {type1}/{aff1} vs {type2}/{aff2}"
                    )
                    continue  # Skip: Bitmasks say they shouldn't collide
                """
                body1 = self.model.geom_bodyid[id1]
                body2 = self.model.geom_bodyid[id2]
                # C. Ignore collisions between two STATIC obstacles
                # Since both are in Group 2, check if both bodies are children of the world (id 0)
                # and have no joints (static).
                if (
                    self.model.body_parentid[body1] == 0
                    and self.model.body_parentid[body2] == 0
                    and self.model.geom_group[id1] == 2
                    and self.model.geom_group[id2] == 2
                ):
                    # This assumes your obstacles are top-level bodies in the worldbody
                    # print(f"Skipping bodies from the same group: {id1}/{id2}")
                    continue

                # B. Check Body Hierarchy (Parent/Child)
                # MuJoCo usually ignores collisions between adjacent links

                if body1 == body2:
                    continue  # Skip: Geoms are on the same body

                # Ignore if body1 is parent of body2 or vice versa
                if (
                    self.model.body_parentid[body1] == body2
                    or self.model.body_parentid[body2] == body1
                ):
                    # print(f"Body {body1} is parent of body {body2} or vice-versa")
                    continue  # Skip: Bodies are directly connected (adjacent links)

                # --- PERFORM ACTUAL COLLISION CHECK ---
                ret = fcl.collide(item1["obj"], item2["obj"], request, result)

                # if (id1 == 1033) or (id2 == 1033):
                drequest = fcl.DistanceRequest()
                dresult = fcl.DistanceResult()

                dret = fcl.distance(item1["obj"], item2["obj"], drequest, dresult)
                # print(
                #    f"Checking collision between object {id1} and {id2} result is {result.is_collision} and ret is {ret} dist {dret}"
                # )
                if result.is_collision:
                    return True

        return False

    def mj_geom_to_fcl(self, model, geom_id):
        """Converts a MuJoCo geom into an FCL collision geometry."""
        g_type = model.geom_type[geom_id]
        size = model.geom_size[geom_id]

        if g_type == mj.mjtGeom.mjGEOM_SPHERE:
            return fcl.Sphere(size[0])
        elif g_type == mj.mjtGeom.mjGEOM_BOX:
            return fcl.Box(size[0] * 2, size[1] * 2, size[2] * 2)
        elif g_type == mj.mjtGeom.mjGEOM_CYLINDER:
            return fcl.Cylinder(size[0], size[1] * 2)  # Radius, Half-length
        elif g_type == mj.mjtGeom.mjGEOM_CAPSULE:
            return fcl.Capsule(size[0], size[1] * 2)  # Radius, Half-length
        elif g_type == mj.mjtGeom.mjGEOM_MESH:
            # 1. Get the mesh ID associated with this geom
            mesh_id = self.model.geom_dataid[geom_id]

            # 2. Get the addresses and counts for vertices and faces
            v_start = self.model.mesh_vertadr[mesh_id]
            v_num = self.model.mesh_vertnum[mesh_id]
            f_start = self.model.mesh_faceadr[mesh_id]
            f_num = self.model.mesh_facenum[mesh_id]

            # 3. Extract vertices and faces
            # mesh_vert is a flat array of [x, y, z, x, y, z...]
            vertices = self.model.mesh_vert[v_start : v_start + v_num]
            # mesh_face is a flat array of [v1, v2, v3, v1, v2, v3...]
            faces = self.model.mesh_face[f_start : f_start + f_num]

            # 4. Create FCL BVH Model
            bvh_model = fcl.BVHModel()
            bvh_model.beginModel(f_num, v_num)
            bvh_model.addSubModel(vertices, faces)
            bvh_model.endModel()

            return bvh_model
        else:
            # Note: For mjGEOM_MESH, you'd need to extract vertices from model.mesh_vert
            raise NotImplementedError(
                f"Geom type {g_type} not implemented for FCL wrapper."
            )

    def add_dynamic_obstacle(
        self, name: str, pos: list, size: list, geom_type=mj.mjtGeom.mjGEOM_BOX
    ):
        """Adds a new obstacle to the spec and recompiles the entire system."""

        # 1. Add to the MjSpec
        new_body = self.spec.worldbody.add_body(name=name, pos=pos)
        new_body.add_geom(
            name=f"geom_{name}",
            type=geom_type,
            size=size,
            rgba=[1, 0, 0, 1],  # Red for dynamic obstacles
            group=2,  # Matching your obstacle group
        )

        # 2. Recompile and refresh
        self.model = self.spec.compile()
        self.data = mj.MjData(self.model)

        # 3. CRITICAL: Refresh FCL objects
        # Clear the old FCL objects and re-allocate based on the new model
        self.fcl_objects = []
        self._init_fcl()

        # 4. Refresh IDs (camera, EEF, etc.) as they may have shifted
        self.eef_site_id = mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, self.end_effector_site_name
        )
        self.camera_id = mj_name2id(
            self.model, mj.mjtObj.mjOBJ_CAMERA, self.CAMERA_NAME
        )
        self.marker_body_ids, self.marker_geom_ids = self._get_marker_ids()

        geom_id = mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, f"geom_{name}")
        print(
            f"Object '{name}' with geomid {geom_id} added. Model recompiled with {self.model.ngeom} geoms."
        )

    def detect_collision_fcl_fast(self) -> bool:
        """Many-to-Many collision detection between Robot and Environment Managers."""

        # 1. Update Robot Transforms
        for item in self.fcl_robot_items:
            g_id = item["geom_id"]
            obj = item["obj"]
            obj.setTransform(
                fcl.Transform(
                    self.data.geom_xmat[g_id].reshape(3, 3), self.data.geom_xpos[g_id]
                )
            )
            self.robot_manager.update(obj)
        self.robot_manager.update()

        # 2. Update Environment Transforms (Required because of your mocap/dynamic obstacles)
        for item in self.fcl_env_items:
            g_id = item["geom_id"]
            obj = item["obj"]
            obj.setTransform(
                fcl.Transform(
                    self.data.geom_xmat[g_id].reshape(3, 3), self.data.geom_xpos[g_id]
                )
            )
            self.env_manager.update(obj)
        self.env_manager.update()

        # 3. Managed Many-to-Many Check
        # Using fcl.defaultCollisionCallback avoids the Python identity/hash overhead
        cdata = fcl.CollisionData()
        self.robot_manager.collide(
            self.env_manager, cdata, fcl.defaultCollisionCallback
        )

        if cdata.result.is_collision:
            return True

        # 4. Optional: Self-Collision
        # If the robot can hit itself, you need a separate check WITH a filtering callback
        # to ignore adjacent links.
        # self.robot_manager.collide(cdata, self._self_collision_callback)

        # check self collision
        if self.detect_self_collision():
            return True

        return False

    def _init_self_collision_pairs(self):
        self.valid_self_collision_pairs = []

        # Iterate through all combinations of robot geoms
        for i in range(len(self.fcl_robot_items)):
            for j in range(i + 1, len(self.fcl_robot_items)):
                item1 = self.fcl_robot_items[i]
                item2 = self.fcl_robot_items[j]

                body1 = self.model.geom_bodyid[item1["geom_id"]]
                body2 = self.model.geom_bodyid[item2["geom_id"]]

                # FILTER LOGIC:
                # 1. Ignore if they belong to the same body
                if body1 == body2:
                    continue

                # 2. Ignore if they are parent/child
                if (
                    self.model.body_parentid[body1] == body2
                    or self.model.body_parentid[body2] == body1
                ):
                    continue

                # 3. Add to whitelist
                self.valid_self_collision_pairs.append((item1, item2))

        print(
            f"Pre-filtered {len(self.valid_self_collision_pairs)} potential self-collision pairs."
        )

    def detect_self_collision(self) -> bool:
        """Pairwise check for the pre-filtered robot links."""
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()

        for item1, item2 in self.valid_self_collision_pairs:
            # FCL collide returns the number of contacts
            if fcl.collide(item1["obj"], item2["obj"], req, res) > 0:
                # Optional: print(f"Self-collision: {item1['name']} with {item2['name']}")
                return True
        return False


# ====================================================================
# OMPL Motion Planning Class
# ====================================================================


class MotionPlannerOMPL:
    """
    A class for generating collision-free robot trajectories using OMPL.
    """

    def __init__(
        self,
        planner_type: str = "RRTConnect",
        collision_method: str = "mujoco",
    ):
        """
        Initializes the OMPL planner structures.
        """
        if not ou:
            raise ImportError(
                "OMPL is not initialized. Please ensure OMPL bindings are installed."
            )

        # Use a dummy path and body name for initialization, will be set later
        self.planner = None
        self.trajectory_buffer = None
        self.planner_type_str = planner_type
        self.collision_method = collision_method
        self.planner_initialized = False
        self.dofs = 0
        self.collison_check_times = []

    def setup_planner(self, robot_planner: RobotPlanner):
        """Initializes OMPL structures using the given RobotPlanner instance."""
        self.robot = robot_planner
        self.dofs = self.robot.model.nq

        # 1. Define the State Space (RealVectorStateSpace for joint angles)
        space = ob.RealVectorStateSpace(self.dofs)

        # 2. Set bounds based on MuJoCo model joint ranges
        bounds = ob.RealVectorBounds(self.dofs)

        """
        for i, jnt_idx in enumerate(self.robot.model.njnt):
            if self.robot.model.jnt_limited[jnt_idx]:
                limit_min = self.robot.model.jnt_range[jnt_idx, 0]
                limit_max = self.robot.model.jnt_range[jnt_idx, 1]
                bounds.setLow(i, limit_min)
                bounds.setHigh(i, limit_max)
            else:
                # Set a generous default bound for unlimited joints (e.g., revolute)
                bounds.setLow(i, -2 * np.pi)
                bounds.setHigh(i, 2 * np.pi)
        """

        # Iterate over all joints (model.njnt) using range() to get indices (jnt_id)
        for jnt_id in range(self.robot.model.njnt):

            # qpos_index is the starting index in the qpos vector for this joint
            qpos_index = int(self.robot.model.jnt_qposadr[jnt_id])

            # Only set bounds for 1-DOF joints (hinge/slide) that are limited
            if self.robot.model.jnt_limited[jnt_id] and self.robot.model.jnt_type[
                jnt_id
            ] in [mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE]:

                limit_min = float(self.robot.model.jnt_range[jnt_id, 0])
                limit_max = float(self.robot.model.jnt_range[jnt_id, 1])

                # Check if the bound index is valid (should be 0 to nq-1)
                if qpos_index < self.dofs:
                    bounds.setLow(qpos_index, limit_min)
                    bounds.setHigh(qpos_index, limit_max)
            else:
                # If unlimited or a complex joint, set a wide range (e.g., -pi to pi for revolute)
                if qpos_index < self.dofs:
                    bounds.setLow(qpos_index, -np.pi)
                    bounds.setHigh(qpos_index, np.pi)

        space.setBounds(bounds)

        # 3. Create Space Information (SI)
        self.si = ob.SpaceInformation(space)

        # 4. Set the State Validity Checker (SVC)
        self.svc = ob.StateValidityCheckerFn(self._is_state_valid)
        self.si.setStateValidityChecker(self.svc)
        self.si.setup()

        # 5. Define the Planner
        self.pdef = ob.ProblemDefinition(self.si)
        self.planner = self._create_planner(self.planner_type_str)

        self.planner_initialized = True

    def _create_planner(self, planner_type: str):
        """Helper to instantiate the chosen OMPL planner."""
        planner_type = planner_type.lower()
        if planner_type == "rrtconnect":
            return og.RRTConnect(self.si)
        elif planner_type == "prm":
            return og.PRM(self.si)
        elif planner_type == "rrtstar":
            return og.RRTstar(self.si)
        else:
            print(
                f"Warning: Planner type '{planner_type}' not recognized. Defaulting to RRTConnect."
            )
            return og.RRTConnect(self.si)

    def _is_state_valid(self, state: ob.State) -> bool:
        """
        Collision checking method used by OMPL.
        Checks: 1) Joint Limits (handled by OMPL bounds, but checked again for robustness).
                 2) MuJoCo Contacts.
        """
        # 1. Convert OMPL state vector to numpy array of actuated joint positions
        q_actuated = np.array([state[i] for i in range(self.dofs)])

        # 2. Check joint limits (optional, as OMPL bounds handle it, but good for custom checks)
        if not self.robot.check_joint_limits(q_actuated):
            return False

        # 3. Apply state to MuJoCo model
        self.robot.set_joint_positions(q_actuated)

        """
        # 4. Check for collision using the chosen method (default: MuJoCo contacts)
        if self.collision_method == "mujoco":
            # detect_collision calls mj_forward internally
            if self.robot.detect_collision():
                return False
        """

        # Use the specific method passed during init
        start = time.perf_counter()
        if self.robot.detect_collision(method=self.collision_method):
            # print("Collision")
            end = time.perf_counter()
            self.collison_check_times.append(end - start)
            return False
        end = time.perf_counter()
        self.collison_check_times.append(end - start)

        # State is valid
        return True

    def plan(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        target_quat: np.ndarray,
        planning_time: float = 5.0,
    ) -> bool:
        self.collison_check_times = []
        """
        Plans a path from a Cartesian start to a Cartesian goal using OMPL.
        """
        if not self.planner_initialized:
            print("Error: Planner not initialized. Call setup_planner first.")
            return False

        self.trajectory_buffer = None
        print("\n--- Phase 1: Finding Start/Goal Joint Angles via IK ---")

        # Find start joint configuration (must start from current state)
        q_start = self.robot.execute_ik_jacobian(start_pos, target_quat)
        if q_start is None:
            print("❌ IK failed for start position. Cannot proceed.")
            return False

        # Find goal joint configuration (using start qpos as initial guess)
        # Note: Must reset qpos to the successful start for the goal IK if needed,
        # but IK uses its own buffer so we just need to ensure the base state is good.
        # We temporarily set the qpos to the start state for the next IK call
        self.robot.set_joint_positions(q_start)
        q_goal = self.robot.execute_ik_jacobian(goal_pos, target_quat)
        if q_goal is None:
            print("❌ IK failed for goal position. Cannot proceed.")
            return False

        # 1. Define Start and Goal States
        start = ob.State(self.si.getStateSpace())
        goal = ob.State(self.si.getStateSpace())

        # OMPL uses RealVectorStateSpace, so we set the joint values
        for i in range(self.dofs):
            start[i] = q_start[i]
            goal[i] = q_goal[i]

        # Check validity of IK solutions
        if not self._is_state_valid(start) or not self._is_state_valid(goal):
            print("❌ Start or goal configuration is invalid (collision or limits).")
            traceback.print_exc()
            return False

        self.pdef.setStartAndGoalStates(start, goal)
        self.planner.setProblemDefinition(self.pdef)
        self.planner.setup()

        print("\n--- Phase 2: OMPL Planning ---")
        print(
            f"Attempting to find path in {planning_time} seconds using {self.planner_type_str}..."
        )

        # 2. Solve the problem
        solved = self.planner.solve(planning_time)

        if solved:
            print("✅ Solution found!")
            path = self.pdef.getSolutionPath()

            # Simplify path (removes redundant intermediate points)
            simplifier = og.PathSimplifier(self.si)
            simplifier.simplify(path, planning_time * 0.5)

            # Convert path to array of joint positions
            num_steps = (
                100  # Interpolate to a fixed number of steps for smooth playback
            )
            path.interpolate(num_steps)

            # Extract and store the joint trajectory
            trajectory = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                q_step = np.array([state[j] for j in range(self.dofs)])
                trajectory.append(q_step)

            self.trajectory_buffer = np.array(trajectory)
            print(
                f"Path simplified and extracted: {len(self.trajectory_buffer)} steps."
            )
            print(
                f"Collision checking per step mean time: {sum(self.collison_check_times)/len(self.collison_check_times)} s max: {max(self.collison_check_times)} min: {min(self.collison_check_times)}"
            )
            return True
        else:
            print("❌ No solution found within the time limit.")
            return False

    def get_trajectory_buffer(self) -> Optional[np.ndarray]:
        """Returns the planned trajectory buffer (N x DOFs)."""
        return self.trajectory_buffer
