import flappy_bird_gymnasium
import gymnasium

from dqn import DQN
from expericenceReplay import ReplayMemory

import itertools
import yaml
import random
import torch
from torch import nn


class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird", render_mode = "human" if render else None, use_lidar = False)
        env = gymnasium.make("CartPole-v1",render_mode = "human" if render else None )

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        

        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            #epsilon greedy
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions)
            target_dqn.load_state_dict(policy_dqn.state_dict())

        for episode in itertools.count():
            state, _ = env.reset()
            # make state into tensor
            state = torch.tensor(state, dtype=torch.float)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                
                #Processing
                new_state, reward, terminated, _, info = env.step(action.item())

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                # accumulate reward
                episode_reward += reward



                if is_training:
                    memory.append(state, action, new_state, reward, terminated)

                #move on to the next state
                state = new_state

            rewards_per_episode.append(episode_reward)
            # Decay epsilon
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=True)