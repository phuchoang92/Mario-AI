import os
import time
import torch
import random
import numpy as np
from itertools import count
from collections import deque
import matplotlib.pyplot as plt
from torchsummary import summary
from Model import DuelingDDQN, DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, policy, save_directory, dueling=False):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.9
        self.current_step = 0
        self.sync_period = 1e4

        # Replay Buffer
        self.batch_size = 32
        self.memory = deque(maxlen=100000)

        self.sync_period = 1e4
        self.training_strategy = policy
        if dueling:
            print("In Dueling Model")
            self.online_model = DuelingDDQN(state_dim, self.action_dim).cuda()
            self.target_model = DuelingDDQN(state_dim, self.action_dim).cuda()
        else:
            print("Not Dueling")
            self.online_model = DQN(state_dim, self.action_dim).cuda()
            self.target_model = DQN(state_dim, self.action_dim).cuda()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=0.00025, eps=1e-4)
        #summary(self.online_model, (4,84,84))
        # path for saing model
        self.save_directory = save_directory

        # for statistic only
        self.episode_reward = []
        self.episode_timestep = []

    def update_target(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((torch.tensor(state.__array__()), torch.tensor(next_state.__array__()),
                            torch.tensor([action], dtype=torch.long), torch.tensor([reward]), torch.tensor([done])))

    def recall(self):
        state, next_state, action, reward, done = map(torch.stack, zip(*random.sample(self.memory, self.batch_size)))
        return state.squeeze(), next_state.squeeze(), action.squeeze(), reward.squeeze(), done.squeeze()

    def act(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, _ = env.step(action)
        self.remember(state, new_state, action, reward, is_terminal)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        return new_state, reward, is_terminal

    def experience_replay(self, model):

        if self.batch_size > len(self.memory):
            return

        state, next_state, action, reward, done = self.recall()

        q_estimate = self.online_model(state.cuda())[np.arange(0, self.batch_size), action.cuda()]
        with torch.no_grad():
            if model == "DDQN":
                best_action = torch.argmax(self.online_model(next_state.cuda()), dim=1)
                next_q = self.target_model(next_state.cuda())[np.arange(0, self.batch_size), best_action]
            if model == "DQN":
                next_q = self.target_model(next_state.cuda()).detach().max(1)[0]
            q_target = (reward.cuda() + (1 - done.cuda().float()) * self.gamma * next_q).float()
        td_error = q_estimate - q_target
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

    def train(self, env, gamma, model_name, max_episodes, checkpoint_period, time_save_model):

        env.seed(42)
        env.action_space.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        # Discounted factor
        self.gamma = gamma
        # for statistic
        self.result = np.empty((max_episodes, 3))
        self.result[:] = np.nan
        training_time = 0
        # Time to save model
        checkpoint_period = checkpoint_period

        self.update_target()

        episode = 0
        while episode != max_episodes:
            episode_start = time.time()

            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)

            while True:
                env.render()
                # time.sleep(0.03)
                state, reward, is_terminal = self.act(state, env)
                self.experience_replay(model_name)

                if np.sum(self.episode_timestep) % self.sync_period == 0:
                    self.update_target()

                if is_terminal:
                    episode += 1
                    if episode % checkpoint_period == 0:
                        self.log_period(episode=episode, step=self.episode_timestep[-1])
                    break

            # statistic
            episode_elapsed = time.time() - episode_start
            training_time += episode_elapsed
            total_step = int(np.sum(self.episode_timestep))
            mean_100_reward = np.mean(self.episode_reward[-100:])
            self.result[episode - 1] = total_step, mean_100_reward, training_time
            if episode % time_save_model == 0:
                self.plot(self.result, episode)
                self.save_checkpoint(episode)

        self.memory.clear()
        # self.plot(self.result)
        return self.result

    def evaluate(self, eval_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            state, done = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                eval_env.render()
                a = self.training_strategy.select_action(eval_model, state)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs)

    def plot(self, result, episode):
        plt.style.use('fivethirtyeight')

        dqn_results = [result]
        dqn_results = np.array(dqn_results)
        ddqn_mean_t, ddqn_mean_r, mean_time = np.mean(dqn_results, axis=0).T

        fig, axs = plt.subplots(2, 1, figsize=(15, 30), sharey=False, sharex=True)
        axs[0].plot(ddqn_mean_r, 'r', label='DQN', linewidth=2)
        axs[0].set_title('Moving Avg Reward (Training)')
        axs[0].legend(loc='upper left')

        plt.xlabel('Episodes')
        fig.savefig(os.path.join(self.save_directory, f"episode_rewards_plot_{episode}.png"))
        '''plt.show()'''

    def log_period(self, episode, step):
        print(f"Episode {episode} - Step {step} - Mean Reward {np.mean(self.episode_reward[-1])}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.online_model.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, episode):
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(episode))
        torch.save(dict(model=self.online_model.state_dict()), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))
