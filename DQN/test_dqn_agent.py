import gym
import keras
import numpy as np
import matplotlib.pyplot as plt
import json


def load_pretrained_model(path):
    """Load a pretrained model from the path provided as parameter"""
    return keras.models.load_model(path)


def select_trained_agent_action(state, trained_model):
    """Uses the trained model to predict the action with highest Q-Value"""
    q_values = trained_model.predict(state, verbose=0)
    return np.argmax(q_values[0])


def plot_epsilon_values():
    """Plots the different values for Epsilon during the training"""
    with open('epsilon_values.txt', 'r') as f:
        file = f.read()
        epsilon_values = json.loads(file)

    plt.plot(range(len(epsilon_values)), epsilon_values)
    plt.show()


def plot_rewards(calculate_mean=1):
    """Plots the rewards obtained during the training. The total number of episodes must be
    a multiple of 'calculate_mean' parameter"""
    with open('rewards.txt', 'r') as f:
        file = f.read()
        rewards = json.loads(file)

    if len(rewards) % calculate_mean != 0:
        raise ValueError

    # Calculate mean of rewards
    mean_rewards = list()
    for i in range(round(len(rewards)/calculate_mean)):
        reward_range = rewards[i*calculate_mean:i*calculate_mean+calculate_mean]
        reward_mean = round(sum(reward_range)/len(reward_range))
        mean_rewards.append(reward_mean)

    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.show()


if __name__ == '__main__':

    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    state_size = env.observation_space.shape[0]
    terminal = False
    trained_agent = load_pretrained_model('trained_agent.h5')

    total_reward = 0
    max_timesteps = 500

    # Execute Episode
    for t in range(max_timesteps):

        env.render()
        state = state.reshape((1, state_size))
        action = select_trained_agent_action(state, trained_agent)
        next_state, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    plot_rewards(1)
    plot_epsilon_values()
