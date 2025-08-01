import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def function_1(x, fixed_point=20):
    return (x-fixed_point)**2


def function_2(x):
    return x**3 + 2*(x**2) - (7*x) + 4


def numerical_integration_derivative(x, function, step_size=0.01):

    # Get X Plus a lil step
    x_step = x + step_size

    # Evaluate Function At X and X_Step
    y_x = function(x)
    y_x_step = function(x_step)

    # Get Difference
    estimated_derivative = (y_x_step - y_x) / step_size

    return estimated_derivative





def newton_raphson_method(function, initial_guess, tolerance=0.01, max_iterations=5000):

    x = initial_guess
    guess_list = []
    for iteration in tqdm(range(max_iterations)):

        # Add Current Guess To List
        guess_list.append(x)

        # Evaluate Function At Initial Guess
        y = function(x)

        # Get Derivative At Initial Guess
        derivative = numerical_integration_derivative(x, function)

        # Get Y Intercept:
        """"
        y = mx + c
        c = y - mx
        """
        c = y - (derivative * x)

        # Get X Intercept
        """
        y = mx + c
        mx = y - c
        x = (y - c) / m        
        """
        x_intercept = (0 - c) / derivative
        print("x intercept", x_intercept)

        if np.abs(x - x_intercept) < tolerance:
            guess_list.append(x_intercept)
            return x_intercept, guess_list

        else:
            x = x_intercept

    print("Error! Did not converge!")
    return None


def plot_guesses(function, guess_list, x_span=[-50, 50], save_directory=None):

    # Visualise Function
    x_values = np.linspace(start=x_span[0], stop=x_span[1], num=1000)
    y_values = []
    for x in x_values:
        y = function(x)
        y_values.append(y)

    plt.ion()
    iteration_count = 0
    figure_1 = plt.figure()
    for guess in guess_list:


        axis_1 = figure_1.add_subplot()
        axis_1.plot(x_values, y_values, c='tab:purple')
        axis_1.scatter(guess, function(guess), c='tab:orange')

        # Hide the right and top spines
        axis_1.spines[['right', 'top']].set_visible(False)
        plt.grid()

        plt.title("Iteration: " + str(iteration_count))
        plt.draw()
        plt.pause(1)

        if save_directory != None:
            plt.savefig(os.path.join(save_directory, str(iteration_count).zfill(3) + ".png"))
        plt.clf()

        iteration_count += 1



fixed_point, guess_list = newton_raphson_method(function_1, initial_guess=-40, tolerance=0.01, max_iterations=5000)
print("fixed_point", fixed_point)
print("Guess list", guess_list)
save_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Dynamical Systems reading group\Week_2_Me\Newton_Raphson\Function_1"
plot_guesses(function_1, guess_list, save_directory=save_directory)



fixed_point, guess_list = newton_raphson_method(function_2, initial_guess=5, tolerance=0.01, max_iterations=5000)
print("fixed_point", fixed_point)
print("Guess list", guess_list)

save_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Dynamical Systems reading group\Week_2_Me\Newton_Raphson\Function_2"
plot_guesses(function_2, guess_list, x_span=[-6, 6], save_directory=save_directory)