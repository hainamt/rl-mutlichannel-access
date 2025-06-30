import copy

from q_table import QTable
import random
from environment import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class QLearnAgent:

    def __init__(self, envi, 
                 episode_length=10000,
                 power_consumption_weight=0.5):
        self.max_reward_episode = None
        self.max_reward = None
        self.max_reward_route = None
        self.env = envi
        num_timesteps = len(self.env)
        num_channels = len(self.env[0])
        self.q_table = QTable(num_timesteps, num_channels)
        self.power_consumption_weight = power_consumption_weight
        self.q_history = []
        self.delta_q = dict()
        self.episode_rewards = []
        self.episode_length = episode_length

    def select_action(self, timestep, current_channel_index, epsilon=0.1):
        random_float = random.random()
        if random_float <= epsilon:
            is_random = True
            action = random.choice(self.q_table.get_all_actions_of_state(timestep, current_channel_index))
        else:
            action = self.q_table.get_best_action(timestep, current_channel_index)
            is_random = False
        # print(f"action: {action}")
        return action, is_random

    def calculate_reward(self, action: Action, next_channel: Channel):
        return next_channel.channel_quality.value - self.power_consumption_weight * action.type.value

    def step(self, timestep: int, current_channel: Channel, action: Action):
        next_channel = current_channel
        next_timestep = timestep + 1
        if next_timestep < len(env):
            for channel in self.env[next_timestep]:
                if channel.channel_index == action.channel_index:
                    next_channel = channel
        return next_channel, self.calculate_reward(action, next_channel)

    def train(self, gamma: float,
               initial_epsilon: float,
               min_epsilon: float,
               epsilon_decay: float,
               initial_learning_rate: float,
               min_learning_rate: float,
               learning_rate_decay: float,
               checkpoints: list[int]):
        max_timestep = len(env) - 1
        epsilon = initial_epsilon
        learning_rate = initial_learning_rate
        
        max_reward = float('-inf')
        max_reward_route = []
        max_reward_episode = -1

        for i in tqdm(range(self.episode_length)):
            rewards = 0
            current_channel = Channel(7)
            epsilon = max(min_epsilon, initial_epsilon * (epsilon_decay ** i))
            learning_rate = max(min_learning_rate, initial_learning_rate * (learning_rate_decay ** i))
            
            episode_route = []

            for timestep in range(len(env)):
                chosen_action, is_random = self.select_action(timestep, current_channel.channel_index, epsilon=epsilon)
                next_channel, reward = self.step(timestep, current_channel, chosen_action)
                rewards += reward
                
                episode_route.append((timestep, current_channel.channel_index, chosen_action))

                old_q = self.q_table.get_q_value(timestep=timestep,
                                            channel_index=current_channel.channel_index,
                                            action=chosen_action)

                if timestep == max_timestep:
                    next_state_best_q = 0
                else:
                    next_timestep = timestep + 1
                    next_state_best_q = self.q_table.get_best_q_value(next_timestep, next_channel.channel_index)

                new_q = old_q + learning_rate * (reward + gamma * next_state_best_q - old_q)
                self.q_table.set_q_value(timestep, current_channel.channel_index, chosen_action, new_q)

                if (timestep, current_channel, chosen_action) not in self.delta_q:
                    self.delta_q[(timestep, current_channel, chosen_action)] = []
                self.delta_q[(timestep, current_channel, chosen_action)].append(abs(new_q - old_q))
                current_channel = next_channel

            self.episode_rewards.append(rewards)
            # print(f"Episode {i}: {rewards}")
            
            if rewards > max_reward:
                max_reward = rewards
                max_reward_route = episode_route.copy()
                max_reward_episode = i
                print(f"New max reward: {max_reward} in episode {i}")

            if i in checkpoints:
                self.q_history.append(copy.deepcopy(self.q_table))

        print(f"Latest epsilon: {epsilon}")
        print(f"Latest learning rate: {learning_rate}")
        print(f"Maximum reward: {max_reward} achieved in episode {max_reward_episode}")
        
        self.max_reward_route = max_reward_route
        self.max_reward = max_reward
        self.max_reward_episode = max_reward_episode


    def draw_best_route(self, start_channel_index: int, ax=None):
        best_route = self.q_table.get_best_route(start_channel_index)
        env_quality = np.array([[item.channel_quality for item in timestep] for timestep in env], dtype=object).T
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True
        else:
            show_plot = False

        color_map = {
            ChannelQuality.BAD: "#f4c7c3",
            ChannelQuality.MEDIUM: "#d9d9d9",
            ChannelQuality.HIGH: "#c6e5c6"
        }

        for i in range(env_quality.shape[0]):
            for j in range(env_quality.shape[1]):
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color_map[env_quality[i, j]], edgecolor='black')
                ax.add_patch(rect)

        for idx in range(len(best_route) - 1):
            t, ch, action = best_route[idx]
            t_next, ch_next, _ = best_route[idx + 1]
            ax.plot([t + 0.5, t_next + 0.5], [ch + 0.5, ch_next + 0.5], 'ro-', linewidth=2)

        t_last, ch_last, _ = best_route[-1]
        ax.plot(t_last + 0.5, ch_last + 0.5, 'ro')

        ax.set_xlim(0, env_quality.shape[1])
        ax.set_ylim(0, env_quality.shape[0])

        ax.set_xticks(np.arange(env_quality.shape[1]) + 0.5)
        ax.set_xticklabels([str(i) for i in range(env_quality.shape[1])])

        ax.set_yticks(np.arange(env_quality.shape[0]) + 0.5)
        ax.set_yticklabels([str(i) for i in range(env_quality.shape[0])])

        ax.set_xlabel('t')
        ax.set_ylabel('Channel index')
        ax.set_title("Best Route")

        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(False)

        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax

    def draw_q_history(self, ax=None):
        average_delta_per_episode = []
        for i in range(self.episode_length):
            episode_deltas = [self.delta_q[(t, c, a)][i] for (t, c, a) in self.delta_q.keys() if i < len(self.delta_q[(t, c, a)])]
            if episode_deltas:
                average_delta_per_episode.append(sum(episode_deltas) / len(episode_deltas))
            else:
                average_delta_per_episode.append(0)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True
        else:
            show_plot = False
        
        ax.plot(average_delta_per_episode)
        ax.set_title('Average Change in Q-Values per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average ΔQ')
        ax.grid(True)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax

    def draw_rewards(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True
        else:
            show_plot = False
        
        ax.plot(self.episode_rewards)
        ax.set_title('Total Reward per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax

    def draw(self, start_channel_index: int):
        fig = plt.figure(figsize=(16, 16))
        gs = plt.GridSpec(3, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])

        self.draw_best_route(start_channel_index, ax=ax1)
        if hasattr(self, 'max_reward_route') and self.max_reward_route:
            self.draw_max_reward_route(ax=ax2)
        self.draw_q_history(ax=ax3)
        self.draw_rewards(ax=ax4)
        
        # Add a convergence plot
        ax5.plot(self.episode_rewards)
        window_size = min(50, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax5.plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2)
        ax5.set_title('Reward Convergence')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Reward')
        ax5.grid(True)
        
        fig.suptitle('Q-Learning Agent Performance Overview', fontsize=16)
        
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))  # Adjust to leave room for the suptitle
        plt.show()
        
        return fig

    def draw_max_reward_route(self, ax=None):
        if not hasattr(self, 'max_reward_route') or not self.max_reward_route:
            print("No maximum reward route recorded. Run training first.")
            return None
            
        env_quality = np.array([[item.channel_quality for item in timestep] for timestep in env], dtype=object).T
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            show_plot = True
        else:
            show_plot = False

        color_map = {
            ChannelQuality.BAD: "#f4c7c3",
            ChannelQuality.MEDIUM: "#d9d9d9",
            ChannelQuality.HIGH: "#c6e5c6"
        }

        for i in range(env_quality.shape[0]):
            for j in range(env_quality.shape[1]):
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color_map[env_quality[i, j]], edgecolor='black')
                ax.add_patch(rect)

        for idx in range(len(self.max_reward_route) - 1):
            t, ch, action = self.max_reward_route[idx]
            t_next, ch_next, _ = self.max_reward_route[idx + 1]
            ax.plot([t + 0.5, t_next + 0.5], [ch + 0.5, ch_next + 0.5], 'go-', linewidth=2)

        t_last, ch_last, _ = self.max_reward_route[-1]
        ax.plot(t_last + 0.5, ch_last + 0.5, 'go')

        ax.set_xlim(0, env_quality.shape[1])
        ax.set_ylim(0, env_quality.shape[0])

        ax.set_xticks(np.arange(env_quality.shape[1]) + 0.5)
        ax.set_xticklabels([str(i) for i in range(env_quality.shape[1])])

        ax.set_yticks(np.arange(env_quality.shape[0]) + 0.5)
        ax.set_yticklabels([str(i) for i in range(env_quality.shape[0])])

        ax.set_xlabel('t')
        ax.set_ylabel('Channel index')
        ax.set_title(f"Max Reward Route (Reward: {self.max_reward:.2f}, Episode: {self.max_reward_episode})")

        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(False)

        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax