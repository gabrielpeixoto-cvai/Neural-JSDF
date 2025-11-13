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


class RobotPlanner:
    """
    A simple MuJoCo class for robot planning with model loading,
    state access, Inverse Kinematics (IK), and collision detection.
    """

    def __init__(self, mjcf_path: str, end_effector_site_name: str):
        """
        1. Loads the robot model using mjspec from an MJCF file.
           The mjspec object is kept for potential future programmatic additions.
        """
        # Load MJCF into MjSpec for programmatic editing
        self.spec = MjSpec.from_file(mjcf_path)

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
            print(
                f"Body {body_id} (Name: {body_name}): Mass = {mass}, Inertia = {inertia}"
            )

        self.end_effector_site_name = end_effector_site_name

        # Get the ID for the end-effector site
        self.eef_site_id = mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, self.end_effector_site_name
        )

        if self.eef_site_id == -1:
            raise ValueError(f"Site '{end_effector_site_name}' not found in model.")

        # --- Joint Identification ---
        # For geometric planning, we plan over all DOFs (model.nq)
        self.num_dofs = self.model.nq

        print(f"RobotPlanner initialized. Planning DOFs (nq): {self.num_dofs}")
        # Call forward once to initialize all data structures
        mj.mj_forward(self.model, self.data)

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

    def detect_collision(self) -> bool:
        """
        4. Detects if there are any active collisions using MuJoCo contacts.
        """
        # mj_step (or mj_forward which is called inside mj_step) populates data.ncon
        # Check if the number of active contacts is greater than 0
        print(self.data.ncon)
        return self.data.ncon > 0

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
        collision_method: str = "mujoco_contacts",
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

        # 4. Check for collision using the chosen method (default: MuJoCo contacts)
        if self.collision_method == "mujoco_contacts":
            # detect_collision calls mj_forward internally
            if self.robot.detect_collision():
                return False

        # State is valid
        return True

    def plan(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        target_quat: np.ndarray,
        planning_time: float = 5.0,
    ) -> bool:
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
            return True
        else:
            print("❌ No solution found within the time limit.")
            return False

    def get_trajectory_buffer(self) -> Optional[np.ndarray]:
        """Returns the planned trajectory buffer (N x DOFs)."""
        return self.trajectory_buffer
