import numpy as np
import matplotlib.pyplot as plt
import os



def leaky_integrate_and_fire(voltage, input_current, equilibrium_voltage, leak_conductance):
    derivative = - leak_conductance * (voltage - equilibrium_voltage) + input_current
    return derivative




def simulate_neuron_no_spiking(model, input_currents, timestep=0.1, n_timepoints=100, equilibrium_voltage=-0.7, leak_conductance=0.5):

    membrane_voltage = []
    current_voltage = equilibrium_voltage

    for timepoint in range(n_timepoints):

        # Add Current Voltage To Trajectory
        membrane_voltage.append(current_voltage)

        # Get Input Current At This Timepoint
        timepoint_input_current = input_currents[timepoint]

        # Get Derivative
        derivative = model(current_voltage, timepoint_input_current, equilibrium_voltage=equilibrium_voltage, leak_conductance=leak_conductance)

        # Multiply By Timestep
        derivative = derivative * timestep

        # Add To Current voltage To Get New Voltage
        new_voltage = current_voltage + derivative

        # Set Current Voltage To New Voltage
        current_voltage = new_voltage

    return membrane_voltage




def simulate_lif_neuron(model,
                        input_currents,
                        timestep=0.1,
                        n_timepoints=100,
                        equilibrium_voltage=-0.7,
                        leak_conductance=0.5,
                        threshold_voltage=-0.55):

    membrane_voltage = np.zeros(n_timepoints)
    spike_times = []

    current_voltage = equilibrium_voltage
    for timepoint in range(n_timepoints):

        # Check If Voltage Above Threshold
        if current_voltage >= threshold_voltage:
            spike_times.append(timepoint) # Record Spike
            current_voltage = current_voltage -0.3  # Reset the neuron by subtracting the threshold voltage

        # Otherwise Compute the voltage using our dynamical system
        elif current_voltage < threshold_voltage:

            # Get Input Current At This Timepoint
            timepoint_input_current = input_currents[timepoint]

            # Get Derivative
            derivative = model(current_voltage, timepoint_input_current, equilibrium_voltage=equilibrium_voltage, leak_conductance=leak_conductance)

            # Multiply By Timestep
            derivative = derivative * timestep

            # Add To Current voltage To Get New Voltage
            new_voltage = current_voltage + derivative

            # Set Current Voltage To New Voltage
            current_voltage = new_voltage

        # Add Current Voltage To Trajectory
        membrane_voltage[timepoint] = current_voltage


    return membrane_voltage, spike_times



def plot_membrane_voltage(voltage, input_current, spike_times=[], threshold_voltage=-0.55):


    figure_1 = plt.figure()
    voltage_axis = figure_1.add_subplot(2, 1, 1)
    input_axis = figure_1.add_subplot(2, 1, 2)


    for time in spike_times:
        voltage[time] = 0.3


    voltage_axis.scatter(spike_times, 0.4 * np.ones(len(spike_times)), c='Grey', marker='s')
    voltage_axis.axhline(threshold_voltage, c='k', linestyle='dashed')

    voltage_axis.plot(voltage)
    input_axis.plot(input_current, c='tab:purple')

    # Hide the right and top spines
    voltage_axis.spines[['right', 'top']].set_visible(False)
    input_axis.spines[['right', 'top']].set_visible(False)


    # Set Labels
    input_axis.set_xlabel("Time")

    voltage_axis.set_ylabel("Membrane Voltage")
    input_axis.set_ylabel("Input Current")

    plt.show()



# Simulation Settings
n_timepoints = 200
timestep = 0.1
equilibrium_voltage = -0.7
leak_conductance= 1.5

# Create Input Current
input_current = np.zeros(n_timepoints)
input_current[5] = 1
for x in range(50, 75, 7):
    input_current[x] = 1.05

for x in range(125, 200, 4):
    input_current[x] = 1


# Run Simulation
membrane_voltage, spike_times = simulate_lif_neuron(leaky_integrate_and_fire, input_current, n_timepoints=n_timepoints,
                                                                            equilibrium_voltage=equilibrium_voltage,
                                                                            leak_conductance=leak_conductance)

plt.plot(membrane_voltage)
plt.show()
print(spike_times)

# Plot Results
plot_membrane_voltage(membrane_voltage, input_current, spike_times)