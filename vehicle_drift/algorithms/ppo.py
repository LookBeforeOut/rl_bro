import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from vehicle_drift.algorithms.base_algorithm import BaseAlgorithm
import os

class PPOPolicy(nn.Module):
    """PPO策略网络"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 动作标准差
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: 观察值
            
        Returns:
            action_mean: 动作均值
            value: 状态价值
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作
        
        Args:
            obs: 观察值
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作
            log_prob: 动作对数概率
            value: 状态价值
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action, log_prob, value

class PPO(BaseAlgorithm):
    """PPO算法实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化PPO算法
        
        Args:
            config: 算法配置参数
        """
        super().__init__(config)
        self.config = config
        
        # 创建策略网络
        self.policy = PPOPolicy(
            obs_dim=config['obs_dim'],
            act_dim=config['act_dim'],
            hidden_dim=config.get('hidden_dim', 64)
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        
        # 超参数
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.target_kl = config.get('target_kl', 0.01)
        self.epochs = config.get('epochs', 10)
        
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据观察值选择动作"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            action, _, _ = self.policy.get_action(obs_tensor, deterministic)
            return action.numpy()
    
    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """更新策略"""
        obs = torch.FloatTensor(batch['obs'])
        act = torch.FloatTensor(batch['act'])
        ret = torch.FloatTensor(batch['ret'])
        adv = torch.FloatTensor(batch['adv'])
        logp_old = torch.FloatTensor(batch['logp'])
        
        metrics = {}
        
        for _ in range(self.epochs):
            # 计算新的动作概率和价值
            _, logp, value = self.policy.get_action(obs)
            ratio = torch.exp(logp - logp_old)
            
            # 计算PPO目标
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = ((value - ret) ** 2).mean()
            
            # 计算熵损失
            entropy_loss = -logp.mean()
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            
            # 更新参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算KL散度
            with torch.no_grad():
                kl = (logp_old - logp).mean()
                if kl > self.target_kl:
                    break
        
        metrics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl': kl.item()
        })
        
        return metrics
    
    def save(self, path: str):
        """保存模型"""
        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
    
    def get_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        return self.config 