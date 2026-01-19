from matplotlib.pyplot import figure
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class multi_armed_bandit_envrioment():

    def step(self, action):

        if action == 0:
            return 1

        elif action == 1:
            return 2


class multi_armed_bandit_agent:

    def __init__(self, epsilon, number_of_actions):

        # Initialise Agent Variables
        self.epsilon = epsilon
        self.number_of_actions = number_of_actions
        self.reward_outcomes = np.zeros(number_of_actions)
        self.action_counters = np.ones(number_of_actions)
        self.q_table = np.zeros(number_of_actions)

    # Agent Action Function
    def select_action(self):
        random_number  = np.random.random()
        if random_number < self.epsilon:
            action = np.random.randint(low=0, high=self.number_of_actions)
        else:
            action = np.argmax(self.q_table)

        return action

    # Update action-value table
    def update_q_table(self, selected_action, reward):
        self.action_counters[selected_action] += 1
        self.reward_outcomes[selected_action] += reward
        self.q_table = np.divide(self.reward_outcomes, self.action_counters)



def run_simulation(n_episodes, epsilon, epsilon_decay):

    reward_gained_list = []
    q_list = []

    # Create Instance of environment
    environment = multi_armed_bandit_envrioment()

    # Create Instance of Agent
    agent = multi_armed_bandit_agent(epsilon=epsilon, number_of_actions=2)

    # Run Through 100 Episodes
    for episode in tqdm(range(n_episodes)):

        # Select Action
        action = agent.select_action()

        # Get Outcome
        outcome = environment.step(action)

        # Update Agent Action Value Table
        agent.update_q_table(action, outcome)

        # Save Total Reward
        reward_gained_list.append(outcome)
        q_list.append(agent.q_table)

        epsilon = epsilon * epsilon_decay

    average_reward = moving_average(reward_gained_list, 5)
    q_list = np.array(q_list)
    return average_reward, q_list


n_episodes = 100
epsilon = 0.1
epsilon_decay = 0.9
average_reward, q_list = run_simulation(n_episodes, 0.1, epsilon_decay)


# Plot Average Reward
figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
axis_1.set_title("Average Reward")
axis_1.plot(average_reward)
axis_1.set_xlabel("Episodes")
axis_1.set_ylabel("Average Reward")
axis_1.spines[['right', 'top']].set_visible(False)
plt.show()


# Plot Q Table Values
figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
axis_1.set_title("Q Table Values")
axis_1.plot(q_list[:, 0], c='b')
axis_1.plot(q_list[:, 1], c='m')
axis_1.set_xlabel("Episodes")
axis_1.set_ylabel("Q Table Values")
axis_1.spines[['right', 'top']].set_visible(False)
plt.show()
