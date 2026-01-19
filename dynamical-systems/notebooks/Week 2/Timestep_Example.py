import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
import os


def dynamical_system(x):
    return x

def simulate_dynamical_system(dynamical_system, initiate_state, n_timespoints=10, timestep=1):

    trajectory = []
    current_state = initiate_state
    for timepoint in range(n_timespoints):

        # Add The Current State To The Trajectory
        trajectory.append(current_state)

        # Get The Derivative at current point
        derivative = dynamical_system(current_state)

        # Move along this derivative by an amount specificed by the timestep
        new_state = current_state + (derivative * timestep)
        current_state = new_state

    return trajectory



def demonstrate_convergence():

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    colourmap = cm.get_cmap("plasma")

    n_iterations = 9
    for x in tqdm(range(1, n_iterations)):

        n_timepoints = 10**x
        timestep = 10 / n_timepoints
        colour_index = float(x) / n_iterations
        colour = colourmap(colour_index)

        trajectory = simulate_dynamical_system(dynamical_system, initiate_state=1, timestep=timestep, n_timespoints=n_timepoints)
        x_values = np.linspace(0, 10, num=n_timepoints)

        axis_1.plot(x_values, trajectory, alpha=0.5, c=colour)

    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    # Set Labels
    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylim([0, 25000])

    plt.title("Timestep: " + str(timestep))

    save_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Dynamical Systems reading group\Week_2_Me\Convergence"
    plt.savefig(os.path.join(save_directory, str(x)+".png"))
    plt.close()



def demonstrate_timestep_effect():

    step_1_trajectory = simulate_dynamical_system(dynamical_system, initiate_state=1, timestep=1, n_timespoints=10)
    step_01_trajectory = simulate_dynamical_system(dynamical_system, initiate_state=1, timestep=0.1, n_timespoints=100)

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Data
    axis_1.plot(np.linspace(0, 10, num=10), step_1_trajectory, alpha=0.5)
    axis_1.scatter(np.linspace(0, 10, num=10), step_1_trajectory, alpha=0.5)

    axis_1.plot(np.linspace(0, 10, num=100), step_01_trajectory, alpha=0.5)
    axis_1.scatter(np.linspace(0, 10, num=100), step_01_trajectory, alpha=0.5)

    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    # Set Labels
    axis_1.set_xlabel("Time (S)")

    plt.show()



demonstrate_timestep_effect()

#demonstrate_convergence()