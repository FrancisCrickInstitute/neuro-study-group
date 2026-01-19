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
        scale = 0.5
        if action == 0:
            #return 1
            return np.random.normal(loc=1, scale=scale)


        elif action == 1:
            return np.random.normal(loc=2, scale=scale)


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
    for episode in range(n_episodes):

        # Select Action
        action = agent.select_action()

        # Get Outcome
        outcome = environment.step(action)

        # Update Agent Action Value Table
        agent.update_q_table(action, outcome)

        # Decrement Epsilon
        agent.epsilon = agent.epsilon * epsilon_decay

        # Save Total Reward
        reward_gained_list.append(outcome)
        q_list.append(agent.q_table)

    #cumulative_reward = np.cumsum(reward_gained_list)
    cumulative_reward =  moving_average(reward_gained_list, 5)

    return cumulative_reward


def run_many_simulations(n_simulations, epsilon, epsilon_decay):

    simulation_outcomes = []
    for x in range(n_simulations):
        reward = run_simulation(200, epsilon, epsilon_decay)
        simulation_outcomes.append(reward)
    simulation_outcomes = np.array(simulation_outcomes)
    return simulation_outcomes



epsilon_values = np.linspace(start=0.1, stop=1, num=10)
epsilon_decay = 0.99
print(epsilon_values)

# Create Figure
figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
axis_1.set_title("Average Reward")
axis_1.set_xlabel("Episodes")
axis_1.set_ylabel("Average Reward")
axis_1.spines[['right', 'top']].set_visible(False)
#axis_1.set_ylim([1, 2.1])

colourmap = plt.get_cmap('plasma')
n_epsilon = len(epsilon_values)

for epsilon_index in tqdm(range(n_epsilon)):
    epsilon = epsilon_values[epsilon_index]
    simulation_outcomes = run_many_simulations(1000, epsilon, epsilon_decay)
    mean_outcome = np.mean(simulation_outcomes, axis=0)
    axis_1.plot(mean_outcome, c=colourmap(epsilon_index / n_epsilon), alpha=0.8)
plt.show()
