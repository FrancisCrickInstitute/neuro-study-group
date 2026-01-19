import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def test_enviroment(id, map_name, q_table):

    enviroment = gym.make(id, map_name=map_name, is_slippery=False, render_mode='human')

    # Reset envrioment
    state = enviroment.reset()[0]
    truncated = False
    terminated = False

    # Run Episode Untill Truncated Or Terminated
    while truncated == False and terminated == False:
            action = np.argmax(q_table[state, :])
            print(truncated)
            new_state, reward, terminated, truncated, _ = enviroment.step(action)
            state = new_state


def train_enviroment(enviroment,
                     q_table,
                     n_episodes,
                     learning_rate,
                     discount_factor,
                     epsilon,
                     epsilon_decay_rate):

    random_number_generator = np.random.default_rng()

    # Iterate Through Each Episode
    episode_mean_reward_list = []
    for iteration in tqdm(range(n_episodes)):

        # Reset envrioment
        state = enviroment.reset()[0]
        truncated = False
        terminated = False
        reward_list = []

        # Run Episode Untill Truncated Or Terminated
        while truncated == False and terminated == False:

            # 1 - Select Action
            # Select Random Action With Some Probability
            random_number  =random_number_generator.random()
            print("random number", random_number)
            if random_number < epsilon:
                action = enviroment.action_space.sample()
                print("action", action)
            # Otherwise Take Best Action In Q Table
            else:
                action = np.argmax(q_table[state, :])

            # 2 - Evaluate Outcome
            new_state, reward, terminated, truncated, _ = enviroment.step(action)
            reward_list.append(reward)

            # 3 - Update Q Table Using Bellman Equation
            q_table_delta = learning_rate * (reward  + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
            q_table[state, action] = q_table[state, action] + q_table_delta

            # set state equal to new state
            state = new_state

        episode_mean_reward =  np.mean(reward_list)
        episode_mean_reward_list.append(episode_mean_reward)

        epsilon = epsilon - epsilon_decay_rate
        if epsilon < 0:
            epsilon = 0

    # Close Enviroment
    enviroment.close()

    return q_table, episode_mean_reward_list



def q_learning_pipeline(id, map_name, n_episodes):

    # Intialise Parameters
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001

    # Create Enviroment
    enviroment = gym.make(id, map_name=map_name, is_slippery=False, render_mode=None)

    # Initialise Q Table
    q_table = np.zeros((enviroment.observation_space.n, enviroment.action_space.n))

    # Train Model
    q_table, rewards_list = train_enviroment(enviroment,
                     q_table,
                     n_episodes,
                     learning_rate,
                     discount_factor,
                     epsilon,
                     epsilon_decay_rate)


    plt.imshow(q_table)
    forceAspect(plt.gca())
    plt.show()

    # Visualise Trained Model
    test_enviroment(id, map_name, q_table)

id = 'FrozenLake-v1'
map_name = "4x4"
q_learning_pipeline(id,map_name, 10000)

