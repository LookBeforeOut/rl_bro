from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseAlgorithm(ABC):
    """强化学习算法基础接口"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """初始化算法
        
        Args:
            config: 算法配置参数
        """
        pass
    
    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """根据观察值选择动作
        
        Args:
            obs: 观察值
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作
        """
        pass
    
    @abstractmethod
    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """更新策略
        
        Args:
            batch: 训练数据批次
            
        Returns:
            metrics: 训练指标
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        pass 