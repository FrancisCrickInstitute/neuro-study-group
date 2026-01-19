import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
ENV_NAME = "FrozenLake-v1"
NUM_EPISODES = 500
MAX_STEPS = 100  # Cap steps to prevent infinite loops early in training

# Hyperparameters
ALPHA = 0.1      # Learning Rate
GAMMA = 0.99     # Discount Factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995 # Decay epsilon every episode

def epsilon_greedy(Q, state, epsilon):
    """
    Selects an action using Epsilon-Greedy strategy.
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, 4) # Explore
    else:
        return np.argmax(Q[state, :])  # Exploit

def run_sarsa():
    # Setup Environment
    # is_slippery=False makes the environment deterministic and learning clearer
    env = gym.make(ENV_NAME, is_slippery=False)
    
    # Initialize Q-Table (16 states x 4 actions)
    action_size = env.action_space.n
    state_size = env.observation_space.n
    Q = np.zeros((state_size, action_size))

    # Metrics storage
    steps_per_episode = []
    epsilon = EPSILON_START

    print(f"Training SARSA on {ENV_NAME} for {NUM_EPISODES} episodes...")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        
        # Choose initial action A (required for SARSA)
        action = epsilon_greedy(Q, state, epsilon)
        
        steps = 0
        done = False
        
        while not done and steps < MAX_STEPS:
            # 1. Take Action A, observe R, S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Modify reward to punish holes slightly and encourage goals
            # (Optional heuristic to speed up graph visualization, 
            # though SARSA works with raw rewards too)
            if terminated and reward == 0: 
                reward = -1 # Fell in hole
            elif terminated and reward == 1:
                reward = 1  # Reached Goal
            elif not done:
                reward = -0.01 # Small penalty for time-wasting
            
            # 2. Choose Action A' from S' using policy
            next_action = epsilon_greedy(Q, next_state, epsilon)
            
            # 3. SARSA Update
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            target = reward + GAMMA * Q[next_state, next_action]
            Q[state, action] += ALPHA * (target - Q[state, action])
            
            # 4. Move to next state/action
            state = next_state
            action = next_action
            steps += 1
            
        steps_per_episode.append(steps)

        # Decay Epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    env.close()
    return steps_per_episode

def plot_learning_curve(steps_data):
    # Calculate a moving average to smooth the plot
    window_size = 20
    moving_avg = np.convolve(steps_data, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    
    # Plot raw data
    plt.plot(steps_data, color='lightblue', alpha=0.5, label='Raw Steps per Episode')
    
    # Plot smoothing curve
    plt.plot(range(len(moving_avg)), moving_avg, color='blue', linewidth=2, label=f'Moving Average ({window_size} eps)')
    
    plt.title('SARSA Learning Curve: Steps to Complete Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    steps_history = run_sarsa()
    plot_learning_curve(steps_history)