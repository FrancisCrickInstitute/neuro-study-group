import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

# --- Configuration ---
NUM_EPISODES = 20
GAMMA = 0.9
PAUSE_TIME = 0.5
OUTPUT_DIR = "mc_frames_ep_return"

# --- Helper Functions ---
def biased_policy(state):
    """Actions: 0:Left, 1:Down, 2:Right, 3:Up"""
    return np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.4, 0.1])

def to_coords(state):
    return state // 4, state % 4

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# --- Visualization Setup ---
def setup_grid_axis(ax, title):
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.set_xticks(np.arange(-0.5, 3.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3.5, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)

def init_plots(env):
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ((ax_traj, ax_val), (ax_n, ax_g)) = axes

    # 1. Trajectory (Top-Left)
    setup_grid_axis(ax_traj, "Current Trajectory")
    ax_traj.set_xlim(-0.5, 3.5)
    ax_traj.set_ylim(3.5, -0.5)
    
    # 2. Value Estimate (Top-Right)
    setup_grid_axis(ax_val, "Global Value Estimate V(s)")
    heatmap_v = ax_val.imshow(np.zeros((4,4)), cmap='Blues', vmin=0, vmax=1.0)
    
    # 3. Visit Counts (Bottom-Left)
    setup_grid_axis(ax_n, "Global Visit Counts N(s)")
    heatmap_n = ax_n.imshow(np.zeros((4,4)), cmap='Oranges') 
    
    # 4. Current Episode Return (Bottom-Right) -- CHANGED
    setup_grid_axis(ax_g, "Return G(s) (Current Episode Only)")
    # We initialize with NaNs so unvisited cells stay white
    init_g = np.full((4,4), np.nan)
    heatmap_g = ax_g.imshow(init_g, cmap='Purples', vmin=0, vmax=1.0) 

    # Static Text & Labels
    desc = env.unwrapped.desc.astype(str)
    text_collections = {'v': [], 'n': [], 'g': []}

    for r in range(4):
        row_v, row_n, row_g = [], [], []
        for c in range(4):
            char = desc[r,c]
            color = 'red' if char == 'H' else 'green' if char == 'G' else 'black'
            
            # Trajectory Background
            ax_traj.text(c, r, char, ha='center', va='center', fontsize=20, 
                         fontweight='bold', color=color, alpha=0.2)
            
            # Subplot Text Placeholders
            for ax, row_list, label in zip([ax_val, ax_n, ax_g], [row_v, row_n, row_g], ["0.00", "0", "-"]):
                # Corner Letter
                ax.text(c - 0.35, r - 0.35, char, ha='center', va='center', 
                        fontsize=8, fontweight='bold', color=color)
                # Central Value
                t = ax.text(c, r, label, ha="center", va="center", 
                            fontsize=11, fontweight='bold', color="black")
                row_list.append(t)
        
        text_collections['v'].append(row_v)
        text_collections['n'].append(row_n)
        text_collections['g'].append(row_g)

    fig.tight_layout()
    return fig, ax_traj, (heatmap_v, heatmap_n, heatmap_g), text_collections, []

def update_trajectory(ax, episode_data, arrows_list):
    for arrow in arrows_list:
        arrow.remove()
    arrows_list.clear()
    
    action_deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    for i, (state, action, reward) in enumerate(episode_data):
        if i < len(episode_data) - 1:
            r, c = to_coords(state)
            dr, dc = action_deltas[action]
            jitter = (np.random.rand() - 0.5) * 0.15
            arrow = ax.arrow(c + jitter, r + jitter, dc*0.5, dr*0.5, 
                             head_width=0.15, head_length=0.15, fc='k', ec='k', width=0.03, alpha=0.6)
            arrows_list.append(arrow)

def update_heatmaps(heatmaps, text_cols, V, N, current_G_grid):
    hm_v, hm_n, hm_g = heatmaps
    
    # Grid conversions
    grid_v = np.zeros((4,4))
    grid_n = np.zeros((4,4))
    
    for s_idx in range(16):
        r, c = to_coords(s_idx)
        grid_v[r,c] = V[s_idx]
        grid_n[r,c] = N[s_idx]

    # Update Images
    hm_v.set_data(grid_v)
    hm_n.set_data(grid_n)
    hm_n.set_clim(vmin=0, vmax=np.max(grid_n) or 1)
    
    # Update G Heatmap (Handles NaNs automatically for white background)
    hm_g.set_data(current_G_grid) 
    
    # Update Text
    for r in range(4):
        for c in range(4):
            # V
            v_val = grid_v[r,c]
            text_cols['v'][r][c].set_text(f"{v_val:.2f}")
            text_cols['v'][r][c].set_color('white' if v_val > 0.6 else 'black')
            
            # N
            n_val = int(grid_n[r,c])
            text_cols['n'][r][c].set_text(f"{n_val}")
            max_n = np.max(grid_n) or 1
            text_cols['n'][r][c].set_color('white' if n_val > 0.6 * max_n else 'black')
            
            # Current G (Check for NaN)
            g_val = current_G_grid[r,c]
            if np.isnan(g_val):
                text_cols['g'][r][c].set_text("-")
                text_cols['g'][r][c].set_color("black")
            else:
                text_cols['g'][r][c].set_text(f"{g_val:.2f}")
                text_cols['g'][r][c].set_color('white' if g_val > 0.6 else 'black')

# --- Main Logic ---
def run_full_viz_mc(env, num_episodes):
    ensure_dir(OUTPUT_DIR)
    S_table = defaultdict(float)
    N_table = defaultdict(int)
    V_table = defaultdict(float)

    fig, ax_traj, heatmaps, texts, arrows = init_plots(env)
    
    print(f"Running {num_episodes} Episodes...")

    for i in range(1, num_episodes + 1):
        # 1. Generate Episode
        episode = []
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = biased_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
            steps += 1
        episode.append((state, None, 0))

        # 2. Compute Returns
        G = 0
        actual_data = episode[:-1]
        
        # Grid to store G ONLY for this episode (filled with NaNs initially)
        current_episode_G = np.full((4,4), np.nan)
        
        for t in range(len(actual_data) - 1, -1, -1):
            state, action, reward = actual_data[t]
            G = GAMMA * G + reward
            
            # Save G to our temporary grid for visualization
            r, c = to_coords(state)
            current_episode_G[r, c] = G
            
            # Update Global Tables (First-Visit Logic)
            previous_states = [x[0] for x in actual_data[:t]]
            if state not in previous_states:
                N_table[state] += 1
                S_table[state] += G
                V_table[state] = S_table[state] / N_table[state]

        # 3. Update Visuals
        outcome = "Goal" if reward == 1 else "Hole"
        ax_traj.set_title(f"Ep {i}: {outcome} | Steps: {len(actual_data)}")
        
        update_trajectory(ax_traj, episode, arrows)
        update_heatmaps(heatmaps, texts, V_table, N_table, current_episode_G)

        fig.canvas.draw()
        fig.canvas.flush_events()
        
        fig.savefig(os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png"))
        time.sleep(PAUSE_TIME)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)
    try:
        run_full_viz_mc(env, NUM_EPISODES)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.close('all')