import mujoco as mj
import numpy as np
import time
from mujoco_planner import RobotPlanner, MotionPlannerOMPL
import traceback

# Assuming the RobotPlanner class is defined in the same file or imported
# (Place the class definition from the previous response above this main block)
# ... [RobotPlanner Class Definition] ...


def main():
    """
    Simple test function for the RobotPlanner class.

    NOTE: You must replace 'path/to/your/robot.xml' and 'tcp_site_name'
          with values corresponding to your specific MJCF file.
    """

    # --- Configuration ---
    MJCF_FILE = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml"  # <--- REPLACE THIS PATH
    EEF_SITE_NAME = "wrist_3_link"  # <--- REPLACE WITH YOUR END-EFFECTOR SITE NAME

    # Example target pose (You will need to adjust these based on your robot's workspace)
    # A common target for IK is just translating the current position slightly.
    # Target orientation (quaternion): [w, x, y, z]. Use [1, 0, 0, 0] for no rotation (identity)
    TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

    # --- Setup ---
    try:
        # 1. Initialize the planner
        planner = RobotPlanner(MJCF_FILE, EEF_SITE_NAME)
        print(f"✅ Model loaded successfully from: {MJCF_FILE}")

    except ValueError as e:
        print(f"❌ Error during initialization: {e}")
        print("Please ensure your MJCF path and end-effector site name are correct.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return

    # --- Pre-IK State ---
    print("\n--- Initial State ---")

    # Ensure kinematics are up-to-date (especially needed before reading TCP position)
    mj.mj_forward(planner.model, planner.data)

    q_initial = planner.get_joint_positions()
    p_initial = planner.get_tcp_position()

    print(
        f"Initial Joint Angles (qpos):\n{q_initial[:7]}"
    )  # Showing first 7 for common robots
    print(f"Initial TCP Position:\n{p_initial}")

    # --- 5. Render the result ---
    print("\n--- Starting Visualization ---")
    # Visualize the robot in the final IK position for 5 seconds
    planner.simulate_and_render(duration=5.0)

    # Define a simple target: move 10 cm in the Z direction (up)
    TARGET_POS = np.array([0.1, 0.1, 0.6])
    print(f"Target TCP Position:\n{TARGET_POS}")

    # --- Execute IK ---
    print("\n--- Running IK Solver ---")

    # 2. Run the IK algorithm
    q_target = planner.execute_ik_jacobian(
        target_pos=TARGET_POS,
        target_quat=TARGET_QUAT,
        max_steps=200,  # Increase steps for more difficult targets
        tolerance=1e-4,  # Higher precision tolerance
    )

    # 3. Apply the results
    planner.data.qpos[:] = q_target

    # Update kinematics after setting new joint angles
    mj.mj_forward(planner.model, planner.data)
    p_final = planner.get_tcp_position()

    # --- Post-IK Results ---
    print("\n--- IK Results ---")
    print(f"Target Joint Angles (qpos):\n{q_target[:7]}")
    print(f"Final TCP Position:\n{p_final}")

    position_error = np.linalg.norm(TARGET_POS - p_final)
    print(f"\nFinal Position Error: **{position_error:.6f}**")

    if position_error < 1e-3:
        print("✅ IK succeeded: The final position is within tolerance (1 mm).")
    else:
        print("⚠️ IK failed to reach the target within the given steps/tolerance.")

    # --- Collision Check ---
    # 4. Check for collision in the final configuration
    print("\n--- Collision Check ---")
    if planner.detect_collision():
        print("❌ Collision detected!")
        contact_info = planner.get_contacts_info()
        print("Contact Details:", contact_info)
    else:
        print("✅ No active collision detected.")
    # --- 5. Render the result ---
    print("\n--- Starting Visualization ---")
    # Visualize the robot in the final IK position for 5 seconds
    planner.simulate_and_render(duration=10.0)


def motion_planning_main():

    # --- Configuration ---
    MJCF_FILE = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml"  # <--- REPLACE THIS PATH
    EEF_BODY_NAME = "wrist_3_link"  # <--- REPLACE WITH YOUR END-EFFECTOR SITE NAME
    PLANNER_TYPE = "RRTConnect"  # Try "PRM" or "RRTstar"

    try:
        # 2. Initialize the Robot Planner
        robot_planner = RobotPlanner(MJCF_FILE, EEF_BODY_NAME)

        # 3. Initialize the Motion Planner
        motion_planner = MotionPlannerOMPL(planner_type=PLANNER_TYPE)
        motion_planner.setup_planner(robot_planner)

    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        traceback.print_exc()
        return

    # --- Define Task ---
    # Current pose (Forward kinematics)
    mj.mj_forward(robot_planner.model, robot_planner.data)
    start_pos = robot_planner.get_tcp_position()
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity orientation

    # Define a goal position (move 20 cm towards the obstacle location)
    # The start position is roughly (0, 0, 0.55). Moving to (0.1, -0.2, 0.2)
    goal_pos = np.array([0.1, 0.1, 0.6])

    print(f"Start Position: {start_pos}")
    print(f"Goal Position: {goal_pos}")

    # --- 4. Plan the Trajectory ---
    # The new goal is designed to be challenging but likely solvable, potentially
    # requiring the robot to maneuver around the red box obstacle at (0.1, 0.1, 0.3).
    success = motion_planner.plan(
        start_pos=start_pos,
        goal_pos=goal_pos,
        target_quat=target_quat,
        planning_time=15.0,  # Increased time for complex planning
    )

    # --- 5. Play Trajectory ---
    if success:
        trajectory = motion_planner.get_trajectory_buffer()
        if trajectory is not None:
            # Play the planned path over 8 seconds total duration
            robot_planner.play_trajectory(trajectory, duration=8.0)
    else:
        print("Cannot play trajectory: Planning failed.")


if __name__ == "__main__":
    # Before running, make sure you have a valid MJCF file and the required
    # MuJoCo Python libraries installed (`pip install mujoco`).
    # main()
    motion_planning_main()
