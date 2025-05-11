import numpy as np
from typing import Dict, Any, Optional
import torch
from vehicle_drift.algorithms.base_algorithm import BaseAlgorithm
from vehicle_drift.envs.base_env import BaseDriftEnv
from torch.utils.tensorboard import SummaryWriter
import os

class Trainer:
    """强化学习训练器"""
    
    def __init__(
        self,
        env: BaseDriftEnv,
        algorithm: BaseAlgorithm,
        config: Dict[str, Any]
    ):
        """初始化训练器
        
        Args:
            env: 环境实例
            algorithm: 算法实例
            config: 训练配置
        """
        self.env = env
        self.algorithm = algorithm
        self.config = config
        
        # 训练参数
        self.max_episodes = config.get('max_episodes', 1000)
        self.max_steps = config.get('max_steps', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        
        # 经验缓冲区
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        
        # 创建TensorBoard写入器
        self.writer = SummaryWriter(log_dir=os.path.join('logs', 'train'))
        
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """计算广义优势估计
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            
        Returns:
            advantages: 优势估计
        """
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(deltas))):
            last_gae = deltas[t] + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
            
        return advantages
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个回合
        
        Returns:
            metrics: 训练指标
        """
        # 重置环境
        obs = self.env.reset()
        episode_reward = 0
        
        # 清空缓冲区
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        
        # 运行回合
        for _ in range(self.max_steps):
            # 选择动作
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action, logp, value = self.algorithm.policy.get_action(obs_tensor)
                action = action.numpy()
                logp = logp.numpy()
                value = value.item()  # 确保value是标量
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.obs_buf.append(obs)
            self.act_buf.append(action)
            self.rew_buf.append(reward)
            self.val_buf.append(value)
            self.logp_buf.append(logp)
            
            # 更新状态
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        # 计算回报和优势
        rewards = np.array(self.rew_buf)
        values = np.array(self.val_buf)
        returns = np.zeros_like(rewards)
        advantages = self.compute_gae(rewards, values)
        
        # 计算回报
        last_value = 0 if done else self.algorithm.policy.critic(torch.FloatTensor(obs)).item()
        returns[-1] = rewards[-1] + self.gamma * last_value
        for t in reversed(range(len(rewards)-1)):
            returns[t] = rewards[t] + self.gamma * returns[t+1]
        
        # 准备训练数据
        batch = {
            'obs': np.array(self.obs_buf),
            'act': np.array(self.act_buf),
            'ret': returns,
            'adv': advantages,
            'logp': np.array(self.logp_buf)
        }
        
        # 更新策略
        metrics = self.algorithm.learn(batch)
        metrics['episode_reward'] = episode_reward
        
        return metrics
    
    def train(self, save_path: Optional[str] = None):
        """训练模型
        
        Args:
            save_path: 模型保存路径
        """
        best_reward = float('-inf')
        
        for episode in range(self.max_episodes):
            metrics = self.train_episode()
            
            # 记录到TensorBoard
            self.writer.add_scalar('Reward/episode_reward', metrics['episode_reward'], episode)
            self.writer.add_scalar('Loss/policy_loss', metrics['policy_loss'], episode)
            self.writer.add_scalar('Loss/value_loss', metrics['value_loss'], episode)
            self.writer.add_scalar('Loss/entropy_loss', metrics['entropy_loss'], episode)
            self.writer.add_scalar('Metrics/kl', metrics['kl'], episode)
            
            # 打印训练信息
            print(f"Episode {episode+1}/{self.max_episodes}")
            print(f"Reward: {metrics['episode_reward']:.2f}")
            print(f"Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"Value Loss: {metrics['value_loss']:.4f}")
            print(f"Entropy Loss: {metrics['entropy_loss']:.4f}")
            print(f"KL: {metrics['kl']:.4f}")
            print("-" * 50)
            
            # 保存最佳模型
            if metrics['episode_reward'] > best_reward and save_path:
                best_reward = metrics['episode_reward']
                self.algorithm.save(save_path)
                print(f"New best model saved with reward: {best_reward:.2f}")
        
        # 关闭TensorBoard写入器
        self.writer.close()
    
    def evaluate(self, num_episodes: int = 10, render: bool = False, video_path: str = None) -> Dict[str, float]:
        """评估模型
        
        Args:
            num_episodes: 评估回合数
            render: 是否渲染环境
            video_path: 视频保存路径，如果为None则不保存视频
            
        Returns:
            metrics: 评估指标
        """
        rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.max_steps):
                action = self.algorithm.act(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

                if render:
                    self.env.render()
                
                if done:
                    break
            
            # 保存轨迹图
            if video_path is not None:
                plot_path = f"{video_path}_episode_{episode+1}.png"
                self.env.save_trajectory_plot(plot_path)
            
            rewards.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        } 