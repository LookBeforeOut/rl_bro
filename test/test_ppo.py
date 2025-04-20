import numpy as np
import gym
from torch import nn, optim
import torch
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button
from bokeh.layouts import column, row
import warnings
import sys

# # 兼容性处理
# if hasattr(np, 'bool8'):
#     np_bool = np.bool8
# else:
#     np_bool = np.bool_  # 新版本numpy使用bool_

# 如果 numpy.bool8 不存在，就让它等于 numpy.bool_
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# 忽略警告
warnings.filterwarnings("ignore")

class PPOTrainer:
    def __init__(self, env_name="CartPole-v1"):
        # 创建环境时指定新版gym API
        self.env = gym.make(env_name, render_mode=None)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim),
            nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.005)
        self.gamma = 0.99
        self.clip_epsilon = 0.2

    def get_action(self, obs):
        """完全兼容的观测值处理"""
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) == 1 else np.array(obs)
        obs = np.array(obs, dtype=np.float32).flatten()
        if not isinstance(obs, np.ndarray):  # 双重保险
            obs = np.array(obs, dtype=np.float32)
        obs = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.policy_net(obs)
        return np.random.choice(self.act_dim, p=probs.detach().numpy()[0])

    def train(self, episodes=200):
        all_rewards = []
        for ep in range(episodes):
            try:
                obs, _ = self.env.reset()  # 新版gym返回两个值
                obs = np.array(obs, dtype=np.float32)

                ep_rewards = []
                states, actions, rewards = [], [], []

                while True:
                    action = self.get_action(obs)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    states.append(obs.copy())
                    actions.append(action)
                    rewards.append(reward)

                    next_obs = np.array(next_obs, dtype=np.float32)
                    obs = next_obs

                    if done:
                        break

                # 计算折扣回报
                discounted_rewards = []
                running_reward = 0
                for r in reversed(rewards):
                    running_reward = r + self.gamma * running_reward
                    discounted_rewards.insert(0, running_reward)

                # 转换为Tensor
                states_tensor = torch.FloatTensor(np.stack(states))
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(discounted_rewards).unsqueeze(1)

                # PPO优化
                with torch.no_grad():
                    old_probs = self.policy_net(states_tensor)
                    old_action_probs = old_probs.gather(1, actions_tensor.unsqueeze(1))

                for _ in range(3):
                    new_probs = self.policy_net(states_tensor)
                    new_action_probs = new_probs.gather(1, actions_tensor.unsqueeze(1))

                    # 重要性采样比率
                    ratios = new_action_probs / old_action_probs

                    # 优势函数估计 (简化版)
                    advantages = rewards_tensor - rewards_tensor.mean()

                    # PPO-Clip目标函数
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    self.optimizer.step()

                ep_reward = sum(rewards)
                all_rewards.append(ep_reward)
                update_plot(ep, ep_reward)

            except Exception as e:
                print(f"第{ep}轮训练出错: {str(e)}")
                break

        return all_rewards

# Bokeh可视化设置
source = ColumnDataSource(data={'episode': [], 'reward': []})
policy_source = ColumnDataSource(data={'x': [], 'y': [], 'action': []})

plot_rewards = figure(width=600, height=300, title="PPO训练曲线")
plot_rewards.line('episode', 'reward', source=source, line_width=2)

plot_policy = figure(width=600, height=300, title="策略可视化 (Cart位置 vs 杆角度)")
plot_policy.scatter('x', 'y', color='action', source=policy_source, alpha=0.6)

button_train = Button(label="开始训练", button_type="success")
is_training = False

def update_plot(ep, reward):
    try:
        new_data = {'episode': [ep], 'reward': [reward]}
        source.stream(new_data)

        if ep % 10 == 0 or ep == 0:
            visualize_policy()
    except Exception as e:
        print(f"更新图表出错: {e}")

def visualize_policy():
    """兼容的策略可视化"""
    try:
        x_vals = np.linspace(-2.4, 2.4, 15)
        y_vals = np.linspace(-0.3, 0.3, 15)
        xx, yy = np.meshgrid(x_vals, y_vals)

        states = np.column_stack([
            xx.ravel(),
            np.zeros_like(xx.ravel()),
            yy.ravel(),
            np.zeros_like(xx.ravel())
        ]).astype(np.float32)

        actions = []
        for s in states:
            try:
                actions.append(trainer.get_action(s))
            except:
                actions.append(0)  # 出错时默认动作

        policy_source.data = {
            'x': states[:, 0],
            'y': states[:, 2],
            'action': [['red', 'blue'][int(a)] for a in actions]  # 确保是整数
        }
    except Exception as e:
        print(f"策略可视化出错: {e}")

def start_training():
    global is_training
    if not is_training:
        is_training = True
        button_train.label = "训练中..."
        button_train.disabled = True

        try:
            trainer.train(episodes=200)
            button_train.label = "训练完成"
        except Exception as e:
            button_train.label = f"错误: {str(e)[:30]}..."
            print(f"训练主循环出错: {e}")
        finally:
            is_training = False
            button_train.disabled = False

button_train.on_click(start_training)

# 初始化
try:
    trainer = PPOTrainer()
    visualize_policy()
except Exception as e:
    print(f"初始化出错: {e}")
    sys.exit(1)

layout = column(
    row(plot_rewards, plot_policy),
    button_train,
    sizing_mode='stretch_width'
)

curdoc().add_root(layout)
curdoc().title = "PPO算法可视化(兼容版)"