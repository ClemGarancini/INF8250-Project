import matplotlib.pyplot as plt

def train_agent(env, agent, num_episodes, batch_size):
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        agent.replay(batch_size)  # Experience replay
        rewards_per_episode.append(total_reward)

        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # Plotting the rewards
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.show()

    return rewards_per_episode
