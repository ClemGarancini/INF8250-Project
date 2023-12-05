import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

def train_dqn_agent(env, agent, num_episodes, batch_size=32):
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_dimensions])  # Reshape for neural network compatibility
        total_rewards = 0

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_dimensions])

            agent.remember(state, action, reward, next_state, done)  # Remember the experience
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

    return rewards_per_episode
