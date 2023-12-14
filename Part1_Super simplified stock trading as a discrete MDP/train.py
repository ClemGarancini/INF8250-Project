import matplotlib.pyplot as plt
import numpy as np

def train_agent(env, agent, num_episodes):
    rewards_per_episode = []
    q_values_array = np.zeros((num_episodes, 11, 6, 6, 2, 2, 4)) 

    for episode in range(num_episodes):
        state_index = env.reset()  # Get the initial state index
        total_rewards = 0

        done = False
        while not done:
            action = agent.choose_action(state_index)
            next_state_index, reward, done = env.step(action)  # next_state_index is directly obtained here
            agent.learn(state_index, action, reward, next_state_index)

            state_ = env.convert_index_to_state(state_index)
            q_values_array[episode, 
                           state_[0], state_[1], state_[2], state_[3], state_[4], 
                           action] = agent.q_table[state_index, action]

            state_index = next_state_index
            total_rewards += reward

        rewards_per_episode.append(total_rewards)
        print(f"Episode: {episode}, Total Reward: {total_rewards}")

    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    return rewards_per_episode, q_values_array
