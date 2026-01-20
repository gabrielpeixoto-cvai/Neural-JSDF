import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import pickle
import time
import mujoco as mj
from mujoco_planner import RobotPlanner, MotionPlannerOMPL
import cv2


# --- 1. MODEL DEFINITION ---
class MiniVLA(nn.Module):
    def __init__(self, action_dim):
        super(MiniVLA, self).__init__()
        # Vision Encoder: Uses a pre-trained ResNet-18 (backbone)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Language Projector: Maps text strings to a fixed-size embedding
        self.lang_embed = nn.Embedding(10, 64)

        # Policy Head: Fuses vision and language features to predict actions
        self.policy = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.ReLU(), nn.Linear(256, action_dim)
        )

    def forward(self, img, text_tokens):
        img_feats = self.backbone(img)  # Output: [Batch, 512]
        text_feats = self.lang_embed(text_tokens).mean(dim=1)  # Output: [Batch, 64]
        combined = torch.cat([img_feats, text_feats], dim=1)
        return self.policy(combined)


# --- 2. DATA UTILITIES ---
class VLADataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        # Minimal vocab mapping for "reach the [color] cube"
        self.vocab = {"reach": 0, "the": 1, "red": 2, "blue": 3, "green": 4, "cube": 5}

    def tokenize(self, text):
        return torch.tensor([self.vocab[w] for w in text.split() if w in self.vocab])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = self.transform(sample["img"]) if self.transform else sample["img"]
        return {
            "img": img,
            "cmd": self.tokenize(sample["cmd"]),
            "action": torch.tensor(sample["action"], dtype=torch.float32),
        }


"""
def get_random_pos_in_fov(planner):
    # 1. Get Camera Parameters from your code logic
    width, height = planner.renderer.width, planner.renderer.height
    fov = planner.model.cam_fovy[planner.camera_id]

    # Calculate focal lengths based on your snippet
    theta = np.deg2rad(fov)
    fx = width / 2 / np.tan(theta / 2)
    fy = height / 2 / np.tan(theta / 2)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # 2. Pick a random pixel in the "center" of the view to avoid edges
    u = np.random.uniform(0.2 * width, 0.8 * width)
    v = np.random.uniform(0.2 * height, 0.8 * height)

    # 3. Create direction in CV frame (Z-forward)
    # This matches the 'intr' matrix logic in your snippet
    dir_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # 4. Transform CV frame to MuJoCo Camera frame (Flip Y and Z)
    # CV: X-right, Y-down, Z-forward | MuJoCo: X-right, Y-up, Z-back
    dir_mj = np.array([dir_cv[0], -dir_cv[1], -dir_cv[2]])

    # 5. Transform from Camera local space to World space
    cam_pos = planner.data.cam_xpos[planner.camera_id]
    cam_mat = planner.data.cam_xmat[planner.camera_id].reshape(3, 3)
    dir_world = cam_mat @ dir_mj

    # 6. Intersect with table plane (Z = 0.02)
    # cam_pos.z + t * dir_world.z = 0.02
    if abs(dir_world[2]) < 1e-6:
        return np.array([0.4, 0.0, 0.02])  # Fallback if ray is parallel

    t = (0.02 - cam_pos[2]) / dir_world[2]

    if t > 0:
        return cam_pos + t * dir_world
    return np.array([0.4, 0.0, 0.02])  # Fallback
"""


def get_random_pos_in_fov(planner, offset=0.1, distance=0.4):
    tcp_pose = planner.get_tcp_position()
    x = np.random.uniform(tcp_pose[0] - offset, tcp_pose[0] + offset)
    y = np.random.uniform(tcp_pose[1] - offset, tcp_pose[1] + offset)
    z = np.random.uniform(tcp_pose[2] - distance - offset, tcp_pose[2] - distance)
    return np.array([x, y, z])


def render_pcl(robot_planner):
    # Capture Point Cloud
    point_cloud_xyz = robot_planner.capture_point_cloud()

    if point_cloud_xyz is not None and point_cloud_xyz.size > 0:

        # Visualize the Point Cloud using remaining markers (Feature 3)
        # Use blue markers for the point cloud
        robot_planner.visualize_geoms(point_cloud_xyz, rgba=np.array([0, 0, 1, 1]))

        # Launch the passive viewer again to see the static point cloud markers
        print("Launching viewer to display static point cloud markers (blue spheres).")
        with mj.viewer.launch_passive(
            robot_planner.model, robot_planner.data
        ) as viewer:
            while viewer.is_running():
                # Display the markers
                viewer.sync()
                time.sleep(0.01)

        # Disable markers after the point cloud viewer closes
        robot_planner.disable_markers()
        mj.mj_forward(robot_planner.model, robot_planner.data)
    else:
        print("Point cloud capture failed or returned no valid points.")


def display_render_img(planner):
    rgb, _ = planner.render_rgbd()
    # print(rgb)
    # CONVERT RGB TO BGR FOR OPENCV
    display_img = rgb.astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("camera", display_img)
    k = cv2.waitKey(1)


# --- 3. MAIN EXECUTION ---
def main():
    # Setup Paths
    MJCF_FILE = "/home/gabriel/robotics_platform/rpf_simulator/robot_models/universal_robots_ur5e/ur5e.xml"  # <--- REPLACE THIS PATH
    EEF_NAME = "wrist_3_link"
    DATA_PATH = "vla_demonstrations.pkl"

    # Initialize Planner
    planner = RobotPlanner(MJCF_FILE, EEF_NAME)
    motion = MotionPlannerOMPL(planner_type="RRTConnect", collision_method="mujoco")
    motion.setup_planner(planner)
    planner.setup_offscreen_rendering(480, 480)

    # --- Define Task ---
    # Current pose (Forward kinematics)
    mj.mj_forward(planner.model, planner.data)
    # planner.simulate_and_render(duration=1000)
    # render_pcl(planner)
    # display_render_img(planner)
    start_pos = planner.get_tcp_position()
    # mujoco uses w,x,y,z
    target_quat = np.array([0.707, -0.707, 0.0, 0.0])  # Identity orientation

    # Define a goal position (move 20 cm towards the obstacle location)
    # The start position is roughly (0, 0, 0.55). Moving to (0.1, -0.2, 0.2)
    goal_pos = np.array([0.4, 0.4, 0.5])

    print(f"Start Position: {start_pos}")
    print(f"Goal Position: {goal_pos}")

    # --- 4. Plan the Trajectory ---
    # The new goal is designed to be challenging but likely solvable, potentially
    # requiring the robot to maneuver around the red box obstacle at (0.1, 0.1, 0.3).
    success = motion.plan(
        start_pos=start_pos,
        goal_pos=goal_pos,
        target_quat=target_quat,
        planning_time=15.0,  # Increased time for complex planning
    )

    # --- 5. Play Trajectory ---
    if success:
        trajectory = motion.get_trajectory_buffer()
        if trajectory is not None:
            # Play the planned path over 8 seconds total duration
            # planner.play_trajectory(trajectory, duration=8.0)
            print("trajectory doen")
    else:
        print("Cannot play trajectory: Planning failed.")

    # --- 7. Sensing and Point Cloud Visualization (Feature 2 & 3) ---

    # Reset to an intermediate pose (or the end pose) to capture the scene
    if success and trajectory is not None:
        reset_q = trajectory[-1]  # Reset to the end of the planned path
    else:
        reset_q = planner.data.qpos.copy()  # Stay at current Q

    planner.set_joint_positions(reset_q)
    mj.mj_forward(planner.model, planner.data)
    print(
        f"TCP pos: {planner.get_tcp_position()} TCP ori: {planner.data.xquat[planner.eef_site_id]}"
    )
    # planner.simulate_and_render(duration=1000)
    # display_render_img(planner)

    # PRE-ALLOCATE CUBES ONCE
    # We add two cubes that we will move around manually
    planner.add_dynamic_obstacle(
        name="target_cube", pos=[0, 0, -1], size=[0.02] * 3, color=[1, 0, 0, 1]
    )
    planner.add_dynamic_obstacle(
        name="distractor_cube", pos=[0, 0, -1], size=[0.02] * 3, color=[0, 0, 1, 1]
    )

    target_geom_id = mj.mj_name2id(
        planner.model, mj.mjtObj.mjOBJ_GEOM, "geom_target_cube"
    )
    distractor_geom_id = mj.mj_name2id(
        planner.model, mj.mjtObj.mjOBJ_GEOM, "geom_distractor_cube"
    )

    """
    # PHASE A: DATA COLLECTION (The "Expert" Script)
    if not os.path.exists(DATA_PATH):
        print("--- Starting Data Collection ---")
        dataset = []
        colors = {"red": [1, 0, 0, 1], "blue": [0, 0, 1, 1]}

        for ep in range(20):  # Collect 20 episodes
            color = np.random.choice(list(colors.keys()))
            target_pos = get_random_pos_in_fov(planner)
            
            #target_pos = [
            #    np.random.uniform(0.2, 0.4),
            #    np.random.uniform(-0.1, 0.1),
            #    0.05,
            #]
            
            print(f"Target pos: {target_pos}")
            planner.add_dynamic_obstacle(
                name=f"cube_{color}_{ep}",
                pos=target_pos,
                size=[0.02] * 3,
                color=colors[color],
            )

            # Expert uses OMPL to find a path
            if motion.plan(
                planner.get_tcp_position(), target_pos, np.array([1, 0, 0, 0])
            ):
                traj = motion.get_trajectory_buffer()
                planner.play_trajectory(traj, duration=8.0)
                for i in range(len(traj) - 1):
                    planner.set_joint_positions(traj[i])
                    rgb, _ = planner.render_rgbd()
                    # Action is the next joint state (Absolute Joint Positions)
                    dataset.append(
                        {
                            "img": rgb,
                            "cmd": f"reach the {color} cube",
                            "action": traj[i + 1],
                        }
                    )
            else:
                planner.simulate_and_render(duration=10)
                print("Invalid trajectory")

        with open(DATA_PATH, "wb") as f:
            pickle.dump(dataset, f)
    """
    planner.release_offscreen_rendering()
    planner.setup_offscreen_rendering(480, 480)

    # PHASE A: DATA COLLECTION
    if not os.path.exists(DATA_PATH):
        print("--- Collecting Data (Reusing Cubes) ---")
        dataset = []

        for ep in range(30):
            # 1. Reset Robot to Home
            # planner.data.qpos[:] = (
            #    planner.model.key_qpos[0] if planner.model.nkey > 0 else 0
            # )
            planner.set_joint_positions(reset_q)
            mj.mj_forward(planner.model, planner.data)

            # 2. Position the existing cubes
            t_pos = get_random_pos_in_fov(planner)
            d_pos = get_random_pos_in_fov(planner)  # Distractor
            print(f"Target pos: {t_pos} Distractor pos: {d_pos}")

            # Move geoms by updating the body positions directly
            t_body_id = planner.model.geom_bodyid[target_geom_id]
            d_body_id = planner.model.geom_bodyid[distractor_geom_id]
            planner.model.body_pos[t_body_id] = t_pos
            planner.model.body_pos[d_body_id] = d_pos

            # Randomly swap colors to teach the model to distinguish
            is_red_target = np.random.random() > 0.5
            if is_red_target:
                planner.model.geom_rgba[target_geom_id] = [1, 0, 0, 1]  # Red
                planner.model.geom_rgba[distractor_geom_id] = [0, 0, 1, 1]  # Blue
                cmd = "reach the red cube"
            else:
                planner.model.geom_rgba[target_geom_id] = [0, 0, 1, 1]  # Blue
                planner.model.geom_rgba[distractor_geom_id] = [1, 0, 0, 1]  # Red
                cmd = "reach the blue cube"
            # planner.simulate_and_render(duration=10)
            # render_pcl(planner)
            # planner.simulate_and_render(duration=10)
            mj.mj_forward(planner.model, planner.data)
            # display_render_img(planner)

            # 3. Expert Plan to target_cube position
            if motion.plan(
                start_pos=planner.get_tcp_position(),
                goal_pos=t_pos + [0.0, 0.0, 0.15],
                target_quat=target_quat,
                planning_time=15.0,  # Increased time for complex planning
            ):
                print("valid trajectory")
                print(f"CMD: {cmd}")
                traj = motion.get_trajectory_buffer()
                # planner.play_trajectory(traj, duration=8.0)
                planner.set_joint_positions(traj[-1])
                mj.mj_forward(planner.model, planner.data)
                print(
                    f"TCP pos: {planner.get_tcp_position()} TCP ori: {planner.data.xquat[planner.eef_site_id]}"
                )
                # planner.simulate_and_render(duration=10)
                for i in range(len(traj) - 1):
                    planner.set_joint_positions(traj[i])
                    mj.mj_forward(planner.model, planner.data)
                    # Calculate the relative change (Delta q)
                    delta_q = traj[i + 1] - traj[i]  #
                    display_render_img(planner)
                    rgb, _ = planner.render_rgbd()
                    # dataset.append({"img": rgb, "cmd": cmd, "action": traj[i + 1]})
                    dataset.append({"img": rgb, "cmd": cmd, "action": delta_q})
            else:
                print("Invalid trajectory")
                print(f"CMD: {cmd}")
                # planner.simulate_and_render(duration=10)

        with open(DATA_PATH, "wb") as f:
            pickle.dump(dataset, f)

    # PHASE B: TRAINING (Behavioral Cloning)
    print("--- Starting Training ---")
    # Use CUDA if available, otherwise check for Mac MPS, else fallback to CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    with open(DATA_PATH, "rb") as f:
        raw_data = pickle.load(f)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((224, 224))]
    )
    vla_data = VLADataset(raw_data, transform=transform)
    loader = DataLoader(vla_data, batch_size=16, shuffle=True)

    model = MiniVLA(action_dim=planner.model.nq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(5):
        for batch in loader:
            pred = model(batch["img"].to(device), batch["cmd"].to(device))
            loss = criterion(pred, batch["action"].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch} Loss: {loss.item():.8f}")

    # PHASE C: INFERENCE & VISUALIZATION
    print("--- Starting Live Action ---")
    planner.set_joint_positions(reset_q)
    mj.mj_forward(planner.model, planner.data)

    planner.add_dynamic_obstacle(
        name="test_cube",
        pos=get_random_pos_in_fov(planner),
        size=[0.02] * 3,
        color=[1, 0, 0, 1],
    )
    test_cmd = vla_data.tokenize("reach the red cube").unsqueeze(0).to(device)

    model.eval()
    planner.release_offscreen_rendering()
    planner.setup_offscreen_rendering(480, 480)

    with mj.viewer.launch_passive(planner.model, planner.data) as viewer:
        for _ in range(200):
            rgb, _ = planner.render_rgbd()
            img_t = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                delta_q = model(img_t, test_cmd).cpu().numpy().flatten()

            current_q = planner.get_joint_positions()
            next_q = current_q + np.clip(delta_q, -0.01, 0.01)
            planner.set_joint_positions(next_q)  # Move to predicted state
            mj.mj_forward(planner.model, planner.data)
            display_render_img(planner)
            viewer.sync()
            time.sleep(0.05)
    planner.release_offscreen_rendering()


if __name__ == "__main__":
    import os

    main()
