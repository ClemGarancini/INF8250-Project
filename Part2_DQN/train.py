import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import torch

def train_dqn_agent(env, agent, num_episodes, batch_size=32, state_bins=10):
    rewards_per_episode = []

    # Define the discretization bins for each state dimension
    balance_bins = np.linspace(0, 20000, state_bins)
    shares_bins = np.linspace(0, 100, state_bins)
    trend_bins = [0, 1]  # Since trend is binary

    q_values_sample = np.zeros((num_episodes, state_bins, state_bins, state_bins, state_bins, state_bins, env.action_space))

    for episode in range(num_episodes):
        state_ = env.reset()
        state = np.reshape(state_, [1, env.state_dimensions])  # Reshape for neural network compatibility
        total_rewards = 0

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_dimensions])

            agent.remember(state, action, reward, next_state, done)  # Remember the experience
            
                        # Discretize the state
            discretized_state = [
                np.digitize(state_[0], balance_bins) - 1,  # Balance
                np.digitize(state_[1], shares_bins) - 1,   # Shares of Pepsi
                np.digitize(state_[2], shares_bins) - 1,   # Shares of Cola
                int(state_[3]),  # Trend of Pepsi
                int(state_[4])   # Trend of Cola
            ]

            # Get Q-values and update the sample array
            state_tensor = torch.from_numpy(state_).float().unsqueeze(0)
            with torch.no_grad():
                q_values = agent.model(state_tensor).numpy()[0]
            q_values_sample[episode, discretized_state[0], discretized_state[1], discretized_state[2], discretized_state[3], discretized_state[4], action] = q_values[action]

            
            state = next_state
            total_rewards += reward

        agent.replay(batch_size)  # Train the model with experiences in memory

        rewards_per_episode.append(total_rewards)
        print(f"Episode: {episode + 1}, Reward: {total_rewards}, Epsilon: {agent.epsilon}")

        # Optionally implement a check for early stopping or model saving

    # Plot the rewards
    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    return rewards_per_episode, q_values_sample
