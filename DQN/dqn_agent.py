import json
import random
import time
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Initialize Replay Buffer as python deque
        self.replay_buffer = deque(maxlen=40000)

        # Set algorithm hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_rate = 10

        # Create both Main and Target Neural Networks
        self.main_network = self.create_nn()
        self.target_network = self.create_nn()

        # Initialize Target Network with Main Network's weights
        self.target_network.set_weights(self.main_network.get_weights())

    def create_nn(self):
        model = Sequential()

        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_network(self):
        """Method to set the Main NN's weights on the Target NN"""
        self.target_network.set_weights(self.main_network.get_weights())

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def sample_experience_batch(self, batch_size):
        # Sample {batchsize} experiences from the Replay Buffer
        exp_batch = random.sample(self.replay_buffer, batch_size)

        # Create an array with the {batchsize} elements for s, a, r, s' and terminal information
        state_batch = np.array([batch[0] for batch in exp_batch]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch]).reshape(batch_size, self.state_size)
        terminal_batch = [batch[4] for batch in exp_batch]

        # Return a tuple, where each item corresponds to each array/batch created above
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def pick_epsilon_greedy_action(self, state):

        # Pick random action with probability ε
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        # Pick action with highest Q-Value (item with highest value for Main NN's output)
        state = state.reshape((1, self.state_size))
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size):

        # Sample a batch of experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_experience_batch(batch_size)

        # Get the actions with highest Q-Value for the batch of next states
        next_q = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q, axis=1)
        # Get the Q-Values of each state in the batch of states
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Update the Q-Value corresponding to the current action with the Target Value
        for i in range(batch_size):
            q_values[i][action_batch[i]] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]

        # Fit the Neural Network
        self.main_network.fit(state_batch, q_values, verbose=0)


if __name__ == '__main__':

    # Initialize CartPole environment
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    # Define state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Define number of episodes, timesteps per episode and batch size
    num_episodes = 150
    num_timesteps = 500
    batch_size = 64
    dqn_agent = DQNAgent(state_size, action_size)
    time_step = 0  # Initialize timestep counter, used for updating Target Network
    rewards, epsilon_values = list(), list()  # List to keep logs of rewards and epsilon values, for plotting later

    for ep in range(num_episodes):

        tot_reward = 0

        state, _ = env.reset()

        print(f'\nTraining on EPISODE {ep+1} with epsilon {dqn_agent.epsilon}')
        start = time.time()

        for t in range(num_timesteps):

            time_step += 1

            # Update Target Network every {dqn_agent.update_rate} timesteps
            if time_step % dqn_agent.update_rate == 0:
                dqn_agent.update_target_network()

            action = dqn_agent.pick_epsilon_greedy_action(state)  # Select action with ε-greedy policy
            next_state, reward, terminal, _, _ = env.step(action)  # Perform action on environment
            dqn_agent.save_experience(state, action, reward, next_state, terminal)  # Save experience in Replay Buffer

            # Update current state to next state and total reward
            state = next_state
            tot_reward += reward

            if terminal:
                print('Episode: ', ep+1, ',' ' terminated with Reward ', tot_reward)
                break

            # Train the Main NN when ReplayBuffer has enough experiences to fill a batch
            if len(dqn_agent.replay_buffer) > batch_size:
                dqn_agent.train(batch_size)

        rewards.append(tot_reward)
        epsilon_values.append(dqn_agent.epsilon)

        # Everytime an episode ends, update Epsilon value to a lower value
        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay

        # Print info about the episode performed
        elapsed = time.time() - start
        print(f'Time elapsed during EPISODE {ep+1}: {elapsed} seconds = {round(elapsed/60, 3)} minutes')

        # If the agent got a reward >499 in each of the last 10 episodes, the training is terminated
        if sum(rewards[-10:]) > 4990:
            print('Training stopped because agent has performed a perfect episode in the last 10 episodes')
            break

    # Save rewards on 'rewards.txt' file
    with open('rewards.txt', 'w') as f:
        f.write(json.dumps(rewards))
    print("Rewards of the training saved in 'rewards.txt'")

    # Save epsilon values
    with open('epsilon_values.txt', 'w') as f:
        f.write(json.dumps(epsilon_values))
    print("Epsilon values of the training saved in 'epsilon_values.txt'")

    # Save trained model
    dqn_agent.main_network.save('trained_agent.h5')
    print("Trained agent saved in 'trained_agent.h5'")

