import numpy as np
import scipy.io as sio
import trimesh
import matplotlib.pyplot as plt


def load_mesh_data_for_viz(file_path="mesh_light_pts.npz", link_index=1):
    """Loads a single mesh structure and its associated points."""
    try:
        if file_path.split(".")[-1] == "mat":
            contents = sio.loadmat(file_path)
            # print(mat_contents)
            # Load the mesh array (assuming 'mesh' is the key)
            mesh_data = contents["mesh"][0]
            print("MESHES")
            print(mesh_data)

            # Select one link (e.g., the second link, index 1)
            # The structure of the arrays is highly nested (confirmed in previous discussion)
            link_struct = mesh_data[link_index]

            # Unpack V and F for Trimesh
            V = link_struct["v"][0][0]
            F = link_struct["f"][0][0] - 1  # Faces must be 0-indexed for Trimesh
            print(V)

            # Unpack pre-sampled points
            int_pts = link_struct["int_pts"][0][0]
            close_pts = link_struct["close_pts"][0][0]
            print(int_pts)

        elif file_path.split(".")[-1] == "npz":
            data = np.load(file_path)
            # Access data for a specific mesh, for example, mesh 0
            V = data[f"mesh_{link_index}_v"]
            F = data[f"mesh_{link_index}_f"] - 1
            int_pts = data[f"mesh_{link_index}_int_pts"]

            # Access data for mesh 5
            close_pts = data[f"mesh_{link_index}_close_pts"]
        else:
            raise NotImplementedError("Invalid format, expects mat or npz files")

        return trimesh.Trimesh(V, F), int_pts, close_pts
    except Exception as e:
        print(f"Error loading data: {e}. Check file path and data structure.")
        return None, None, None


def visualize_single_link_points(link_index=1):
    """Visualizes a single link mesh, its interior, and close-to-surface points."""
    mesh, int_pts, close_pts = load_mesh_data_for_viz(link_index=link_index)

    if mesh is None:
        return

    # 1. Mesh Geometry
    # mesh_geom = trimesh.visual.create.icosphere(
    #    subdivisions=2, color=[0.5, 0.5, 0.5, 0.5]
    # )
    # mesh_geom.apply_scale(0.001)  # Example scaling
    mesh.visual.face_colors = [150, 150, 150, 150]

    # 2. Interior Points (e.g., Red)
    int_pc = trimesh.points.PointCloud(int_pts, colors=[255, 0, 0, 255])

    # 3. Close Points (e.g., Blue)
    close_pc = trimesh.points.PointCloud(close_pts, colors=[0, 0, 255, 255])

    # Combine all into a scene
    scene = trimesh.Scene([mesh, int_pc, close_pc])

    # Launch the visualization window
    print("Launching Trimesh viewer. Red = Interior points, Blue = Close points.")
    scene.show()


# Example usage (uncomment to run locally):
visualize_single_link_points(link_index=4)
