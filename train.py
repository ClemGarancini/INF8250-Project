import matplotlib.pyplot as plt

def train_agent(env, agent, num_episodes):
    rewards_per_episode = []

    for episode in range(num_episodes):
        state_index = env.reset()  # Get the initial state index
        total_rewards = 0

        done = False
        while not done:
            action = agent.choose_action(state_index)
            next_state_index, reward, done = env.step(action)  # next_state_index is directly obtained here
            print(f"next_state_index: {next_state_index}, type: {type(next_state_index)}")
            agent.learn(state_index, action, reward, next_state_index)
            state_index = next_state_index
            total_rewards += reward

        rewards_per_episode.append(total_rewards)
        print(f"Episode: {episode}, Total Reward: {total_rewards}")

    plt.plot(rewards_per_episode)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    return rewards_per_episode
