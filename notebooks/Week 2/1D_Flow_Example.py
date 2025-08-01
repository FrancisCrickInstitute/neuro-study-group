import numpy as np
import os
import matplotlib.pyplot as plt



def simplest_dynamical_system(x):
    return x

def negative_dynamical_system(x):
    return -x


def dynamical_system_specific_equilibirum(x, c=2):
    return -(x-c)


def dynamical_system_leakage_current(x, c=2, g=4):
    return -g * (x-c)


def simulate_1d_dynamical_system(system, timestep=0.1, n_trajectories=5, n_timepoints=10, initial_conditions=None, initial_condition_range=[-1,1]):

    if initial_conditions == None:
        initial_conditions = np.random.uniform(low=initial_state_range[0], high=initial_state_range[1], )

    trajectory_list = []
    for trajectory_index in range(n_trajectories):
        trajectory = []

        current_state = initial_conditions[trajectory_index]
        for timepoint_index in range(n_timepoints):
            trajectory.append(current_state)
            derivative = system(current_state)
            new_state = np.add(current_state, derivative * timestep)
            current_state = new_state

        trajectory_list.append(trajectory)

    trajectory_list = np.array(trajectory_list)
    return trajectory_list




def plot_vector_field(function, x_range=[-2,2], y_range=[-2,2], x_density=100, y_density=5):

    # Create Lists To Span The Range of X and Y
    x_values = np.linspace(start=x_range[0], stop=x_range[1], num=x_density)
    y_values = np.linspace(start=y_range[0], stop=y_range[1], num=y_density)

    x_state_list = []
    y_state_list = []
    x_derivative_list = []
    y_derivative_list = []

    for x_value in x_values:
        for y_value in y_values:

            # Get Derivative
            [x_derivative, y_derivative] = function(x_value)

            # Add To List
            x_state_list.append(x_value)
            y_state_list.append(y_value)
            x_derivative_list.append(x_derivative)
            y_derivative_list.append(y_derivative)

    # Create Figure
    figure_1 = plt.figure()

    # Plot Vector Field Using Quiver
    axis_1 = figure_1.add_subplot()
    axis_1.quiver(x_state_list, y_state_list, x_derivative_list, y_derivative_list)


    # Move Axes To Centre
    axis_1.spines['left'].set_position('center')
    axis_1.spines['bottom'].set_position('center')
    axis_1.spines['right'].set_color('none')
    axis_1.spines['top'].set_color('none')
    axis_1.xaxis.set_ticks_position('bottom')
    axis_1.yaxis.set_ticks_position('left')


    plt.show()





def plot_1d_vector_field(function, x_range=[-2,2], x_density=100, ylim=[-1,1]):

    # Create Lists To Span The Range of X and Y
    x_values = np.linspace(start=x_range[0], stop=x_range[1], num=x_density)
    y_values = np.zeros(x_density)

    x_state_list = []
    y_state_list = []
    x_derivative_list = []
    y_derivative_list = []

    for x_value in x_values:
        for y_value in y_values:

            # Get Derivative
            x_derivative = function(x_value)
            y_derivative = 0

            # Add To List
            x_state_list.append(x_value)
            y_state_list.append(y_value)
            x_derivative_list.append(x_derivative)
            y_derivative_list.append(y_derivative)

    # Create Figure
    figure_1 = plt.figure()

    # Plot Vector Field Using Quiver
    axis_1 = figure_1.add_subplot()
    axis_1.quiver(x_state_list, y_state_list, x_derivative_list, y_derivative_list, color="tab:purple", clip_on=False, zorder=5, scale_units='xy', units='xy')

    # Move Axes To Centre
    axis_1.spines['left'].set_position('center')
    axis_1.spines['bottom'].set_position('center')
    axis_1.spines['right'].set_color('none')
    axis_1.spines['top'].set_color('none')
    axis_1.xaxis.set_ticks_position('bottom')
    axis_1.yaxis.set_ticks_position('left')

    axis_1.set_xlim(x_range)
    axis_1.set_ylim(ylim)

    plt.show()






def plot_1d_vector_field_with_trajectories(function, x_range=[-2,2], x_density=100, ylim=[-1,1]):

    # Create Lists To Span The Range of X and Y
    x_values = np.linspace(start=x_range[0], stop=x_range[1], num=x_density)
    y_values = np.zeros(x_density)

    x_state_list = []
    y_state_list = []
    x_derivative_list = []
    y_derivative_list = []

    for x_value in x_values:
        for y_value in y_values:
            # Get Derivative
            x_derivative = function(x_value)
            y_derivative = 0

            # Add To List
            x_state_list.append(x_value)
            y_state_list.append(y_value)
            x_derivative_list.append(x_derivative)
            y_derivative_list.append(y_derivative)

    # Create Figure
    figure_1 = plt.figure()

    # Plot Vector Field Using Quiver
    axis_1 = figure_1.add_subplot()
    axis_1.quiver(x_state_list, y_state_list, x_derivative_list, y_derivative_list, color="tab:purple", clip_on=False, zorder=5, angles='xy')

    # Move Axes To Centre
    axis_1.spines['left'].set_position('center')
    axis_1.spines['bottom'].set_position('center')
    axis_1.spines['right'].set_color('none')
    axis_1.spines['top'].set_color('none')
    axis_1.xaxis.set_ticks_position('bottom')
    axis_1.yaxis.set_ticks_position('left')

    axis_1.set_xlim(x_range)
    axis_1.set_ylim(ylim)

    plt.show()








#plot_1d_vector_field(simplest_dynamical_system, x_range=[-4,4], x_density=9)
#plot_1d_vector_field(negative_dynamical_system, x_range=[-4,4], x_density=9)
#plot_1d_vector_field(dynamical_system_specific_equilibirum, x_range=[-5,5], x_density=9)
plot_1d_vector_field(dynamical_system_leakage_current, x_range=[-5,5], x_density=9)
