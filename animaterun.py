import pandas as pd
import matplotlib.pyplot as plt
import glob
import plotly.graph_objects as go

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

def plot_total_mass_over_time(total_mass_data):
    """
    Plot the total mass of the cluster over time.

    Parameters:
    - total_mass_data: A DataFrame with 'Simulation Time (Myr)' and 'Total Mass (MSun)' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        total_mass_data["Simulation Time (Myr)"],
        total_mass_data["Total Mass (MSun)"],
        marker="o",
        color="blue",
        linestyle="-",
        label="Total Mass",
    )
    plt.xlabel("Simulation Time (Myr)")
    plt.ylabel("Total Mass (M☉)")
    plt.ylim(bottom=0)
    plt.title("Total Mass of the Cluster Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mass_over_time.png")
    plt.show()


def create_3d_animation(data):
    """
    Create a 3D interactive animation of star positions over time.
    
    Parameters:
    - data: A DataFrame containing star positions and simulation time.
    """
    # Calculate axis ranges
    x_min, x_max = data["X (pc)"].min(), data["X (pc)"].max()
    y_min, y_max = data["Y (pc)"].min(), data["Y (pc)"].max()
    z_min, z_max = data["Z (pc)"].min(), data["Z (pc)"].max()

    # Extract unique time steps
    timesteps = sorted(data["Simulation Time (Myr)"].unique())
    frames = []

    for t in timesteps:
        # Filter data for the current timestep
        frame_data = data[data["Simulation Time (Myr)"] == t]
        scatter = go.Scatter3d(
            x=frame_data["X (pc)"],
            y=frame_data["Y (pc)"],
            z=frame_data["Z (pc)"],
            mode="markers",
            marker=dict(
                size=3,
                color=frame_data["Mass (MSun)"],
                colorscale="Viridis",
                colorbar=dict(
                    title="Mass (M☉)",
                    tickvals=[data["Mass (MSun)"].min(), data["Mass (MSun)"].max()],
                ),
                opacity=0.8
            ),
            name=f"Time {t}"  # This will show up in the legend
        )
        frames.append(go.Frame(data=[scatter], name=f"Time {t}"))

    # Initial scatter plot for the first timestep
    initial_data = data[data["Simulation Time (Myr)"] == timesteps[0]]
    scatter = go.Scatter3d(
        x=initial_data["X (pc)"],
        y=initial_data["Y (pc)"],
        z=initial_data["Z (pc)"],
        mode="markers",
        marker=dict(
            size=3,
            color=initial_data["Mass (MSun)"],
            colorscale="Viridis",
            colorbar=dict(
                title="Mass (M☉)",
                tickvals=[data["Mass (MSun)"].min(), data["Mass (MSun)"].max()],
            ),
            opacity=0.8
        ),
        name="Stars"
    )

    # Define layout with fixed axes
    layout = go.Layout(
        title="3D Animation of Star Positions",
        scene=dict(
            xaxis=dict(title="X (pc)", range=[x_min, x_max]),
            yaxis=dict(title="Y (pc)", range=[y_min, y_max]),
            zaxis=dict(title="Z (pc)", range=[z_min, z_max]),
        ),
        plot_bgcolor="black",  # Set the plot background to black
        paper_bgcolor="black",  # Set the outer paper background to black
        font=dict(color="white"),  # Set text to white for better visibility
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            None,
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[  # Add a slider to control animation
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(font=dict(size=20), prefix="Time: ", visible=True, xanchor="right"),
                steps=[
                    dict(
                        label=f"Time {t}",
                        method="animate",
                        args=[[f"Time {t}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                    )
                    for t in sorted(data["Simulation Time (Myr)"].unique())
                ],
            )
        ]
    )

    # Create figure
    fig = go.Figure(data=[scatter], layout=layout, frames=frames)

    # Show animation
    fig.show()
    fig.write_html("star_positions_animation.html")

# Example usage
directory = "./csv1"
data = load_csv_data(directory)

#Calculate total mass at each time step
total_mass_data = calculate_total_mass(data)

# Plot the change in total mass over time
plot_total_mass_over_time(total_mass_data)

# Make animation
create_3d_animation(data)