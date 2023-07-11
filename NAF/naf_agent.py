import random
import time
from collections import deque, namedtuple
from typing import Optional, Any, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_


class NAF(nn.Module):

    def __init__(self, state_size: int, action_size: int, layer_size: int, seed: int, device: torch.device) -> None:
        """
        Model to be used in the NAF algorithm. Network Architecture:
        - Common network
            - Linear + BatchNormalization (input_shape, layer_size)
            - Linear + BatchNormalization (layer_size, layer_size)

        - Output for mu network (used for calculating A)
            - Linear (layer_size, action_size)

        - Output for V network (used for calculating Q = A + V)
            - Linear (layer_size, 1)

        - Output for L network (used for calculating P = L . Lt)
            - Linear (layer_size, (action_size*action_size+1)/2)
        Args:
            state_size: Dimension of a state.
            action_size: Dimension of an action.
            layer_size: Size of the hidden layers of the neural network.
            seed: Random seed.
            device: CUDA device.
        """
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # DEFINE THE MODEL
        # Define the first NN hidden layer + BatchNormalization
        self.input_layer = nn.Linear(in_features=self.state_size, out_features=layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)

        # Define the second NN hidden layer + BatchNormalization
        self.hidden_layer = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)

        # Define the output layer for the mu Network
        self.action_values = nn.Linear(in_features=layer_size, out_features=action_size)
        # Define the output layer for the V Network
        self.value = nn.Linear(in_features=layer_size, out_features=1)
        # Define the output layer for the L Network
        self.matrix_entries = nn.Linear(in_features=layer_size,
                                        out_features=int(self.action_size * (self.action_size + 1) / 2))

    def forward(self,
                input_: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Any], Any]:
        """
        Forward propagation.
        It feeds the NN with the input, and gets the output for the mu, V and L networks.
        - Output from the L network is used to create the P matrix.
        - Output from the V network is used to calculate the Q value: Q = A + V
        - Output from the mu network is used to calculate A. The action output of mu nn is considered
            the action that maximizes Q-function.
        Args:
            input_: Input for the neural network's input layer.
            action: Current action, used for calculating the Q-Function estimate.
        Returns:
            Returns a tuple containing the action which maximizes the Q-Function, the
            Q-Function estimate and the Value Function.
        """
        # FEED INPUT DATA TO THE NEURAL NETWORK
        # Feed the input to the INPUT_LAYER and apply ReLu activation function (+ BatchNorm)
        x = torch.relu(self.bn1(self.input_layer(input_)))
        # Feed the output of INPUT_LAYER to the HIDDEN_LAYER layer and apply ReLu activation function (+ BatchNorm)
        x = torch.relu(self.bn2(self.hidden_layer(x)))
        # Feed the output of HIDDEN_LAYER to the mu layer and apply tanh activation function
        action_value = torch.tanh(self.action_values(x))
        # Feed the output of HIDDEN_LAYER to the L layer and apply tanh activation function
        matrix_entries = torch.tanh(self.matrix_entries(x))
        # Feed the output of HIDDEN_LAYER to the V layer
        V = self.value(x)
        # Modifies the output of the mu layer by unsqueezing it (all tensor as a 1D vector)
        action_value = action_value.unsqueeze(-1)

        # CREATE L MATRIX from the outputs of the L layer
        # Create lower-triangular matrix, size: (n_samples, action_size, action_size)
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(self.device)
        # Get lower triagular indices (returns list of 2 elems, where the first row contains row coordinates
        # of all indices and the second row contains column coordinates)
        lower_tri_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)
        # Fill matrix with the outputs of the L layer
        L[:, lower_tri_indices[0], lower_tri_indices[1]] = matrix_entries
        # Raise the diagonal elements of the matrix to the square
        L.diagonal(dim1=1, dim2=2).exp_()
        # Calculate state-dependent, positive-definite square matrix P
        P = L * L.transpose(2, 1)

        # CALCULATE Q-VALUE
        Q = None
        if action is not None:
            # Calculate Advantage Function estimate
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P),
                                     (action.unsqueeze(-1) - action_value))).squeeze(-1)
            # Calculate Q-values
            Q = A + V

        # ADD NOISE TO ACTION
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)

        return action, Q, V


class ReplayBuffer:

    def __init__(self, buffer_size: int, batch_size: int, device: torch.device, seed: int):
        """
        Buffer to store experience tuples. Each experience has the following structure:
        (state, action, reward, next_state, done)
        Args:
            buffer_size: Maximum size for the buffer. Higher buffer size imply higher RAM consumption.
            batch_size: Number of experiences to be retrieved from the ReplayBuffer per batch.
            device: CUDA device.
            seed: Random seed.
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state: NDArray, action: NDArray, reward: float, next_state: NDArray, done: int) -> None:
        """
        Add a new experience to the Replay Buffer.
        Args:
            state: NDArray of the current state.
            action: NDArray of the action taken from state {state}.
            reward: Reward obtained after performing action {action} from state {state}.
            next_state: NDArray of the state reached after performing action {action} from state {state}.
            done: Integer (0 or 1) indicating whether the next_state is a terminal state.
        """
        # Create namedtuple object from the experience
        exp = self.experience(state, action, reward, next_state, done)
        # Add the experience object to memory
        self.memory.append(exp)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.
        Returns:
            Tuple of 5 elements, which are (states, actions, rewards, next_states, dones). Each element
            in the tuple is a torch Tensor composed of {batch_size} items.
        """
        # Randomly sample a batch of experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state if not isinstance(e.state, tuple) else e.state[0] for e in experiences])).float().to(
            self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current size of the Replay Buffer
        Returns:
            Size of Replay Buffer
        """
        return len(self.memory)


class NAFAgent:

    def __init__(self,
                 environment: gym.Env,
                 state_size: int,
                 action_size: int,
                 layer_size: int,
                 batch_size: int,
                 buffer_size: int,
                 learning_rate: float,
                 tau: float,
                 gamma: float,
                 update_freq: int,
                 num_updates: int,
                 device: torch.device,
                 seed: int) -> None:
        """
        Interacts with and learns from the environment via the NAF algorithm.
        Args:
            environment: Instance of gym.Env class (OpenAI gym environment).
            state_size: Dimension of the states.
            action_size: Dimension of the actions.
            layer_size: Size for the hidden layers of the neural network.
            batch_size: Number of experiences to train with per training batch.
            buffer_size: Maximum number of experiences to be stored in Replay Buffer.
            learning_rate: Learning rate for neural network's optimizer.
            tau: Hyperparameter for soft updating the target network.
            gamma: Discount factor.
            update_freq: Number of timesteps after which the main neural network is updated.
            num_updates: Number of updates performed when learning.
            device: Device used (CPU or CUDA).
            seed: Random seed.
        """
        self.environment = environment
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        random.seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.batch_size = batch_size

        # Initalize Q-Networks
        self.qnetwork_main = NAF(state_size, action_size, layer_size, seed, device).to(device)
        self.qnetwork_target = NAF(state_size, action_size, layer_size, seed, device).to(device)

        # Define Adam as optimizer
        self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr=learning_rate)

        # Initialize Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device, seed)

        # Initialize update time step counter (for updating every {update_freq} steps)
        self.update_t_step = 0

    def step(self, state: NDArray, action: NDArray, reward: float, next_state: NDArray, done: int) -> None:
        """
        Stores in the ReplayBuffer the new experience composed by the parameters received,
        and learns only if the Buffer contains enough experiences to fill a batch. The
        learning will occur if the update frequency {update_freq} is reached, in which case it
        will learn {num_updates} times.
        Args:
            state: Current state.
            action: Action performed from state {state}.
            reward: Reward obtained after performing action {action} from state {state}.
            next_state: New state reached after performing action {action} from state {state}.
            done: Integer (0 or 1) indicating whether a terminal state have been reached.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learning will be performed every {update_freq}} time-steps.
        self.update_t_step = (self.update_t_step + 1) % self.update_freq  # Update time step counter
        if self.update_t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    # Pick random batch of experiences from memory
                    experiences = self.memory.sample()

                    # Learn from experiences and get loss
                    self.learn(experiences)

    def act(self, state: NDArray) -> NDArray:
        """
        Extracts the action which maximizes the Q-Function, by getting the output of the mu layer
        of the main neural network.
        Args:
            state: Current state from which to pick the best action.
        Returns:
            Action which maximizes Q-Function.
        """
        state = torch.from_numpy(state).float().to(self.device)

        # Set evaluation mode on naf_components for obtaining a prediction
        self.qnetwork_main.eval()
        with torch.no_grad():
            # Get the action with maximum Q-Value from the local network
            action, _, _ = self.qnetwork_main(state.unsqueeze(0))

        # Set training mode on naf_components for future use
        self.qnetwork_main.train()

        return action.cpu().squeeze().numpy()

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """
        Calculate the Q-Function estimate from the main neural network, the Target value from
        the target neural network, and calculate the loss with both values, all by feeding the received
        batch of experience tuples to both networks. After loss is calculated, backpropagation is performed on the
        main network from the given loss, so that the weights of the main network are updated.
        Args:
            experiences: Tuple of five elements, where each element is a torch.Tensor of length {batch_size}.
        """
        # Set gradients of all optimized torch Tensors to zero
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # Get the Value Function for the next state from target naf_components (no_grad() disables gradient calculation)
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute the target Value Functions for the given experiences.
        # The target value is calculated as target_val = r + gamma * V(s')
        target_values = rewards + (self.gamma * V_)

        # Compute the expected Value Function from main network
        _, q_estimate, _ = self.qnetwork_main(states, actions)

        # Compute loss between target value and expected Q value
        loss = F.mse_loss(q_estimate, target_values)

        # Perform backpropagation for minimizing loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_main.parameters(), 1)
        self.optimizer.step()

        # Update the target network softly with the local one
        self.soft_update(self.qnetwork_main, self.qnetwork_target)

    def soft_update(self, main_nn: NAF, target_nn: NAF) -> None:
        """
        Soft update naf_components parameters following this formula:
                    θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            main_nn: Main torch neural network.
            target_nn: Target torch neural network.
        """
        for target_param, main_param in zip(target_nn.parameters(), main_nn.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1. - self.tau) * target_param.data)

    def run(self, mean_reward_period: int, frames: int = 1000, episodes: int = 1000) -> None:
        """
        Execute training flow of the NAF algorithm on the given environment.
        Args:
            mean_reward_period: the plot which shows the rewards will show the mean reward every
            {mean_reward_period} episodes.
            frames: Number of maximum frames or timesteps per episode.
            episodes: Number of episodes required to terminate the training.
        Returns:
            Returns the score history generated along the training.
        """
        rewards = list()
        # Iterate through every episode
        for episode in range(episodes):
            state, _ = self.environment.reset()
            rewards.append(0.)
            print(f'\n\nRunning episode {episode+1}\n')

            for frame in range(1, frames + 1):

                # Pick action according to current state
                action = self.act(state)

                # Perform action on environment and get new state and reward
                next_state, reward, done, _, _ = self.environment.step(np.resize(action, (1,)))

                # Save the experience in the ReplayBuffer, and learn from previous experiences if applicable
                self.step(state, action, reward, next_state, done)

                state = next_state  # Update state to next state
                rewards[-1] += reward

                if done:
                    break

            print(f'Episode {episode+1} finished after {frame} steps')
            print(f'Reward obtained for episode {episode+1}: {rewards[-1]}')

        # Show the mean reward obtained every <mean_reward_period> episodes
        mean_rewards = [sum(rewards[i*mean_reward_period:i*mean_reward_period+mean_reward_period])/mean_reward_period
                        for i in range((len(rewards) // mean_reward_period) - 1)]

        plt.plot(range(len(mean_rewards)), mean_rewards)
        plt.show()


def evaluate_trained_model(environment: gym.Env, agent: NAFAgent,
                           n_episodes: int, frames: int) -> None:
    """
    Tests a previously trained agent through the execution of {n_episodes} test episodes,
    for {frames} timesteps each. When the test concludes, the results of the test are logged on terminal.
    Args:
        environment: Gym Environment.
        agent: Previously trained NAFAgent instance.
        n_episodes: Number of test episodes to execute.
        frames: Number of timesteps per test episode.
    """
    # Initialize Test result's history
    results = list()

    for ep in range(n_episodes):
        state, _ = environment.reset()

        for frame in range(frames):
            action = agent.act(state)
            next_state, reward, done, _, _ = environment.step(np.resize(action, (1,)))
            state = next_state
            if done:
                if reward > 90:
                    results.append((True, frame))
                    break
                else:
                    results.append((False, frame))
                    break

            if frame == frames - 1 and not done:
                results.append((False, frame))
                break

        print(f'Test Episode number {ep+1} completed\n')

    print('RESULTS OF THE TEST:')
    for i, result in enumerate(results):
        print(f'Results of Iteration {i + 1}: COMPLETED: {result[0]}. FRAMES: {result[1]}')

    print(f'Number of successful executions: '
          f'{[res[0] for res in results].count(True)}/{len(results)}  '
          f'({([res[0] for res in results].count(True) / len(results)) * 100}%)')
    print(f'Average number of frames required to complete an episode: '
          f'{np.mean(np.array([res[1] for res in results]))}')


if __name__ == '__main__':
    # Create the environment and train the agent
    env = gym.make('MountainCarContinuous-v0')
    naf_agent = NAFAgent(environment=env,
                         state_size=2,
                         action_size=1,
                         layer_size=64,
                         batch_size=32,
                         buffer_size=10000,
                         learning_rate=0.001,
                         tau=0.001,
                         gamma=0.99,
                         update_freq=1,
                         num_updates=1,
                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                         seed=314)
    start = time.time()
    naf_agent.run(mean_reward_period=5, frames=1500, episodes=1000)
    print(f'\n\nTraining took {time.time() - start} seconds\n\n')

    # Evaluate the agent
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    evaluate_trained_model(env, naf_agent, 5, 1000)
