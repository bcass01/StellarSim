import pandas as pd
import matplotlib.pyplot as plt
import glob

def load_csv_data(directory):
    """
    Load star position data from all CSV files in a directory.
    
    Parameters:
    - directory: Path to the directory containing CSV files.
    
    Returns:
    - A DataFrame containing all data.
    """
    csv_files = sorted(glob.glob(f"{directory}/*.csv"))  # Sort files by timestep
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(dataframes, ignore_index=True)
    return all_data

def calculate_total_mass(data):
    """
    Calculate the total mass of the cluster at each time step.

    Parameters:
    - data: A DataFrame containing star data with columns including 'Mass (MSun)' and 'Simulation Time (Myr)'.

    Returns:
    - A DataFrame with 'Simulation Time (Myr)' and 'Total Mass (MSun)' columns.
    """
    total_mass_by_time = (
        data.groupby("Simulation Time (Myr)")["Mass (MSun)"]
        .sum()
        .reset_index()
        .rename(columns={"Mass (MSun)": "Total Mass (MSun)"})
    )
    return total_mass_by_time

def plot_multiple_mass_series(data_list, labels):
    """
    Plot the total mass over time for multiple datasets.

    Parameters:
    - data_list: A list of DataFrames, each containing 'Simulation Time (Myr)' and 'Total Mass (MSun)' columns.
    - labels: A list of labels corresponding to each dataset.
    """
    plt.figure(figsize=(12, 8))

    for data, label in zip(data_list, labels):
        plt.plot(
            data["Simulation Time (Myr)"],
            data["Total Mass (MSun)"],
            marker="o",
            markersize=1,
            linestyle="-",
            linewidth=0.5,
            label=label,
        )

    plt.xlabel("Simulation Time (Myr)")
    plt.ylabel("Total Mass (M☉)")
    plt.title("Total Mass of the Cluster Over Time (Comparison)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mass_comparison_over_time.png")
    plt.show()

# Example usage
directories = [f"./r{i:02}/csv1" for i in range(1, 31)]  # Assuming 30 directories named csv1, csv2, ..., csv30
data_list = []
labels = []

for i, directory in enumerate(directories, start=1):
    data = load_csv_data(directory)
    total_mass_data = calculate_total_mass(data)
    data_list.append(total_mass_data)
    labels.append(f"Dataset {i}")  # Label each dataset for the legend

# Plot all datasets
plot_multiple_mass_series(data_list, labels)
