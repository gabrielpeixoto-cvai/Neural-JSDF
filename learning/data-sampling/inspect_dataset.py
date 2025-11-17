import numpy as np
import scipy.io as sio  # Conventionally imported as sio
import os


def load_and_inspect_mat_file(filepath):
    """Loads a .mat file and prints its top-level keys."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at path: {filepath}")
        return

    print(f"Loading file: {filepath}...")

    # Use loadmat to load the data into a dictionary
    mat_contents = sio.loadmat(filepath)

    print("\n--- Top-Level Keys (MATLAB Variable Names) ---")
    # MATLAB variables like __header__, __version__, __globals__ are automatically added
    for key in mat_contents.keys():
        if not key.startswith("__"):
            print(
                f"Key: '{key}', Type: {type(mat_contents[key])}, Shape: {mat_contents[key].shape}"
            )

    print("\n--- Inspecting First User Variable ---")
    # Assuming the first user variable is the one you want to inspect (e.g., 'mesh')

    first_key = next(
        (key for key in mat_contents.keys() if not key.startswith("__")), None
    )

    if first_key:
        print(f"Inspecting variable '{first_key}':")
        data = mat_contents[first_key]

        # NOTE: MATLAB structs and cell arrays often come out as multi-dimensional
        # NumPy arrays with dtype=object or structured arrays.

        # Example from your gen_dataset.py:
        # mat_contents = sio.loadmat("meshes/mesh_light_pts.mat")
        # mesh = mat_contents["mesh"][0]

        if data.dtype == np.dtype("object") or data.dtype.fields is not None:
            print(
                "⚠️ Data is complex (e.g., MATLAB struct/cell array). Accessing inner content may require indexing."
            )
            # If it's a 1xN structured array (common for structs/cell arrays)
            if data.ndim > 1 and data.shape[0] == 1:
                print(
                    f"Try accessing the first element: mat_contents['{first_key}'][0]"
                )

        # If the data is simply an array
        print(f"Data dtype: {data.dtype}")
        print(f"Data shape: {data.shape}")
        if data.size > 0:
            for i in range(10):
                print(
                    f"Joints: {data[i,:7]}\nPoint: {data[i,7:10]}\ndistances: {data[i,10:]}\n"
                )
    return data


def calculate_and_compare_stats(data_py, data_mat):
    """Calculates min, max, mean, and compares them."""

    if data_py is None or data_mat is None:
        print("\nCannot proceed. One or both datasets failed to load.")
        return

    # Extract Distance Columns (assuming distances start at column index 10)
    try:
        dist_py = data_py[:, 10:]
        dist_mat = data_mat[:, 10:]
    except IndexError:
        print(
            "\nFATAL ERROR: Column count is too small. Check if your array has distances starting at index 10."
        )
        print(
            f"Python data columns: {data_py.shape[1]}, MATLAB data columns: {data_mat.shape[1]}"
        )
        return

    N_MESHES = dist_py.shape[1]

    print(f"\n--- Statistical Comparison of {N_MESHES} Distance Columns ---")

    # Calculate Statistics (using the absolute value of the distances for cleaner comparison of scale)
    stats_py = {
        "min": np.min(np.abs(dist_py)),
        "max": np.max(np.abs(dist_py)),
        "mean": np.mean(np.abs(dist_py)),
        "std": np.std(np.abs(dist_py)),
    }

    stats_mat = {
        "min": np.min(np.abs(dist_mat)),
        "max": np.max(np.abs(dist_mat)),
        "mean": np.mean(np.abs(dist_mat)),
        "std": np.std(np.abs(dist_mat)),
    }

    print(
        f"\n{'Metric':<10}{'Python (Meters)':<20}{'MATLAB (Meters)':<20}{'Ratio (PY/MAT)':<15}"
    )
    print("-" * 65)

    for metric in ["min", "max", "mean", "std"]:
        ratio = (
            stats_py[metric] / stats_mat[metric] if stats_mat[metric] != 0 else np.inf
        )

        print(
            f"{metric:<10}{stats_py[metric]:<20.6f}{stats_mat[metric]:<20.6f}{ratio:<15.2f}"
        )

    print("-" * 65)

    # Conclusion on Unit
    if 90 < (stats_py["mean"] / stats_mat["mean"]) < 110:
        print(
            f"\nConclusion: The **Python dataset** is likely in **Centimeters** and the MATLAB dataset is in Meters (or vice versa). The mean ratio is close to 100."
        )
    elif 0.9 < (stats_py["mean"] / stats_mat["mean"]) < 1.1:
        print(
            f"\nConclusion: The units match! Both datasets are likely in **Meters** (ratio is close to 1)."
        )
    else:
        print(
            f"\nConclusion: The units do not match, and the difference is not a simple factor of 100. There may be another error."
        )


# You need to replace this with the path to your MATLAB data file:
mat_data = load_and_inspect_mat_file("datasets/data_mesh_test.mat")
data = np.load("robot_dataset_py.npy")
calculate_and_compare_stats(data, mat_data)
