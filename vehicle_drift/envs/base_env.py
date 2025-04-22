from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any

class BaseDriftEnv(ABC):
    """基础车辆漂移环境接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """初始化环境
        
        Args:
            config: 环境配置参数
        """
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境到初始状态
        
        Returns:
            observation: 初始观察值
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作
        
        Args:
            action: 动作向量
            
        Returns:
            observation: 新的观察值
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human'):
        """渲染环境
        
        Args:
            mode: 渲染模式
        """
        pass
    
    @abstractmethod
    def close(self):
        """关闭环境"""
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Dict:
        """返回观察空间"""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Dict:
        """返回动作空间"""
        pass 