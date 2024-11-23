from amuse.lab import units, Hermite
from amuse.ic.kroupa import new_kroupa_mass_distribution
from amuse.ic.kingmodel import new_king_model
from amuse.units import nbody_system
from amuse.community.seba.interface import SeBa
from amuse import config
config.channel_type = "mpi"  # Force MPI as the communication channel
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

class StarClusterSimulation:
    def __init__(self, N, t_end, dt, diagnostic_interval=1|units.Myr, 
                 Mmin=1.0|units.MSun, Mmax=100.0|units.MSun, z=0.02):
        self.N = N
        self.t_end = t_end
        self.dt = dt
        self.diagnostic_interval = diagnostic_interval
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.z = z
        
        # Storage for evolution data
        self.times = []
        self.kinetic_energies = []
        self.potential_energies = []
        self.cumulative_wall_times = []
        self.stellar_type_changes = []
        
        # Storage for final HR diagram data
        self.final_temperatures = []
        self.final_luminosities = []
        self.final_radii = []
        self.final_types = []

    def initialize_gravity(self, w0):
        """Initialize gravitational dynamics using a King model."""
        self.converter = nbody_system.nbody_to_si(self.N | units.MSun, 1 | units.parsec)
        self.stars = new_king_model(self.N, w0, self.converter)
        self.stars.mass = new_kroupa_mass_distribution(self.N, 
                                                       mass_min=0.1 | units.MSun, 
                                                       mass_max=125.0 | units.MSun)
        self.gravity = Hermite(self.converter)
        self.gravity.particles.add_particles(self.stars)
        self.gravity_to_stars = self.gravity.particles.new_channel_to(self.stars)

    def initialize_stellar(self):
        """Initialize stellar evolution."""
        self.stellar = SeBa()
        self.stellar.parameters.metallicity = self.z
        self.stellar.particles.add_particles(self.stars)
        self.stellar_to_gravity = self.stellar.particles.new_channel_to(self.gravity.particles)

    def compute_truncation_radius_based_on_density(self, density_threshold=1e-5 | (units.MSun / units.parsec**3)):
        distances = self.stars.position.lengths()
        
        # Safeguard for empty star array
        if len(distances) == 0:
            print("No stars remaining in the cluster.")
            return 0 | units.parsec

        max_radius = distances.max()
        
        # Use numpy linspace with numerical values and reapply units
        for r in np.linspace(0.9 * max_radius.value_in(units.parsec), max_radius.value_in(units.parsec), 100) | units.parsec:
            density = self.get_density_at_radius(r)
            if density < density_threshold:
                return r
        return max_radius

    def get_density_at_radius(self, radius):
        """Calculate density at a given radius."""
        mask = self.stars.position.lengths() <= radius
        mass_in_sphere = self.stars[mask].mass.sum()
        volume_of_sphere = (4/3) * np.pi * (radius**3)
        density = mass_in_sphere / volume_of_sphere
        return density

    def remove_escaped_stars(self, truncation_radius):
        """Remove stars beyond the truncation radius."""
        # Select stars with distances greater than the truncation radius
        escaped_stars = self.stars.select(lambda position: position.length() > truncation_radius, ["position"])
        if len(escaped_stars) > 0:
            print(f"{len(escaped_stars)} stars escaped the cluster.")
            self.gravity.particles.remove_particles(escaped_stars)
            self.stars.remove_particles(escaped_stars)

    def plot_star_distances_and_truncation(self, step, truncation_radius, sim_time):
        """
        Plot star distances from the cluster center with truncation radius.

        Parameters:
        - step: The current timestep index (used for file naming).
        - truncation_radius: The truncation radius at the current timestep.
        - sim_time: The simulation time.
        """
        distances = self.stars.position.lengths().value_in(units.parsec)
        truncation_radius_value = truncation_radius.value_in(units.parsec)

        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(distances)), distances, s=5, label="Star Distances")
        plt.axhline(y=truncation_radius_value, color='red', linestyle='--', label=f"Truncation Radius ({truncation_radius_value:.2f} pc)")
        plt.xlabel("Star Index")
        plt.ylabel("Distance from Cluster Center (pc)")
        plt.title(f"Time: {sim_time.value_in(units.Myr):.2f} Myr")  # Fixed formatting issue
        plt.legend()
        plt.grid(True)

        # Save the plot with the timestep index
        #plt.savefig(f"cluster_step_{step:05d}.png")
        plt.close()


    def evolve(self, density_threshold=1e-5 | (units.MSun / units.parsec**3)):
        """Evolve the cluster and track total mass."""
        time_start = time.time()
        sim_time = 0 | units.Myr
        truncation_radii = []  # Store truncation radii for analysis
        self.total_masses = []  # Store total mass at each timestep

        try:
            step = 0
            while sim_time < self.t_end:
                sim_time += self.dt

                # Evolve gravity and stellar systems
                self.gravity.evolve_model(sim_time)
                self.gravity_to_stars.copy()
                self.stellar.evolve_model(sim_time)
                self.stellar_to_gravity.copy()

                # Write star data to CSV
                self.write_star_data_to_csv(step, sim_time)

                # Compute and apply truncation radius
                truncation_radius = self.compute_truncation_radius_based_on_density(density_threshold)
                truncation_radii.append(truncation_radius)
                print(f"Truncation radius at {sim_time.in_(units.Myr)}: {truncation_radius.value_in(units.parsec):.2f} pc")

                
                # Plot star distances and truncation radius
                self.plot_star_distances_and_truncation(step, truncation_radius, sim_time)
                
                # Remove escaped stars
                self.remove_escaped_stars(truncation_radius)

                # Record total cluster mass
                total_mass = self.stars.mass.sum()
                self.total_masses.append(total_mass)

                step += 1

            print("Simulation complete.")
        finally:
            self.gravity.stop()
            self.stellar.stop()

    def plot_density_profile(self):
        """Plot the cluster density profile."""
        distances = self.stars.position.lengths().value_in(units.parsec)
        plt.hist(distances, bins=30, density=True)
        plt.xlabel("Radius (pc)")
        plt.ylabel("Density")
        plt.title("Density Profile")
        plt.savefig("density_profile.png")  # Save as PNG
        plt.close()  # Close the plot to prevent display issues


    def plot_total_mass(self):
        """
        Plot the total mass of the cluster over time.
        """
        if not hasattr(self, 'total_masses') or len(self.total_masses) == 0:
            print("No total mass data to plot.")
            return
        
        times = np.linspace(0, self.t_end.value_in(units.Myr), len(self.total_masses))
        masses = [mass.value_in(units.MSun) for mass in self.total_masses]

        plt.figure(figsize=(8, 6))
        plt.plot(times, masses, label="Total Mass", color="blue", linewidth=2)
        plt.xlabel("Time (Myr)")
        plt.ylabel("Total Mass (M☉)")
        plt.title("Total Mass of the Cluster Over Time")
        plt.grid(True)
        plt.legend()
        # Save the graph as a file
        plt.savefig("total_mass_over_time.png")  # Save as PNG
        plt.close()

    def write_star_data_to_csv(self, step, sim_time):
        """
        Write the masses and positions of stars to a CSV file for the current timestep.

        Parameters:
        - step: The current timestep index.
        - sim_time: The simulation time.
        """
        filename = f"star_data_step_{step:05d}.csv"
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["Star_ID", "Mass (MSun)", "X (pc)", "Y (pc)", "Z (pc)", "Simulation Time (Myr)"])

            # Write data for each star
            for i, star in enumerate(self.stars):
                mass = star.mass.value_in(units.MSun)
                position = star.position.value_in(units.parsec)
                writer.writerow([i, mass, position[0], position[1], position[2], sim_time.value_in(units.Myr)])



def main():
    N = 2000
    t_end = 1000 | units.Myr
    dt = 0.1 | units.Myr
    w0 = 3
    density_threshold = 1e-5 | (units.MSun / units.parsec**3)

    # Initialize and run simulation
    sim = StarClusterSimulation(N=N, t_end=t_end, dt=dt)
    sim.initialize_gravity(w0=w0)
    sim.initialize_stellar()
    sim.evolve(density_threshold=density_threshold)
    sim.plot_density_profile()
    sim.plot_total_mass()

if __name__ == "__main__":
    main()
