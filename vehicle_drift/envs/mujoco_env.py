import mujoco
import numpy as np
from typing import Dict, Tuple, Any
import os
from vehicle_drift.envs.base_env import BaseDriftEnv

class MujocoDriftEnv(BaseDriftEnv):
    """Mujoco车辆漂移环境实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Mujoco环境
        
        Args:
            config: 环境配置参数
        """
        super().__init__(config)
        self.config = config
        
        # 加载Mujoco模型
        model_path = os.path.join(os.path.dirname(__file__), '..', config['model_path'])
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 设置观察空间和动作空间
        self._observation_space = {
            'shape': (13,),  # 位置(3) + 姿态(4) + 速度(3) + 角速度(3)
            'dtype': np.float32
        }
        self._action_space = {
            'shape': (2,),  # [steering, throttle]
            'low': np.array([-1.0, 0.0]),
            'high': np.array([1.0, 1.0]),
            'dtype': np.float32
        }
        
        # 初始化渲染器
        self.renderer = None
        # if config.get('render_mode') == 'human':
        #     self.renderer = mujoco.Renderer(self.model)
        
        # 控制参数
        self.control_freq = config.get('control_freq', 50)
        self.sim_freq = config.get('sim_freq', 500)
        self.sim_steps = int(self.sim_freq / self.control_freq)
        
    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始位置和速度
        self.data.qpos[0:3] = [0, 0, 0.5]  # 位置
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 姿态（四元数）
        self.data.qvel[0:3] = [0, 0, 0]  # 线速度
        self.data.qvel[3:6] = [0, 0, 0]  # 角速度
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        # 应用动作
        self.data.ctrl[0] = action[0] * 0.5  # 转向 [-0.5, 0.5]
        self.data.ctrl[1] = action[1] * 100  # 油门 [0, 100]
        
        # 前向仿真
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
        
        # 获取观察值
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self._check_done()
        
        # 额外信息
        info = {
            'velocity': np.linalg.norm(self.data.qvel[0:3]),
            'drift_angle': self._calculate_drift_angle(),
            'position': self.data.qpos[0:3].copy(),
            'orientation': self.data.qpos[3:7].copy()
        }
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if self.renderer is not None:
            self.renderer.update_scene(self.data)
            self.renderer.render()
    
    def close(self):
        """关闭环境"""
        if self.renderer is not None:
            self.renderer.close()
    
    @property
    def observation_space(self) -> Dict:
        return self._observation_space
    
    @property
    def action_space(self) -> Dict:
        return self._action_space
    
    def _get_obs(self) -> np.ndarray:
        """获取观察值"""
        # 位置 (3)
        position = self.data.qpos[0:3]
        
        # 姿态 (4) - 四元数
        orientation = self.data.qpos[3:7]
        
        # 速度 (3)
        velocity = self.data.qvel[0:3]
        
        # 角速度 (3)
        angular_velocity = self.data.qvel[3:6]
        
        return np.concatenate([position, orientation, velocity, angular_velocity])
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        # 获取当前状态
        velocity = np.linalg.norm(self.data.qvel[0:3])
        drift_angle = self._calculate_drift_angle()
        
        # 计算奖励
        reward = (
            1.0 * abs(drift_angle) +  # 漂移角度权重
            0.1 * velocity  # 速度权重
        )
        
        # 添加控制惩罚
        reward -= 0.01 * (  # 控制惩罚权重
            abs(self.data.ctrl[0]) + abs(self.data.ctrl[1])
        )
        
        # 检查碰撞
        if self._check_collision():
            reward -= 10.0  # 碰撞惩罚
        
        return reward
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        # 检查碰撞
        if self._check_collision():
            return True
        
        # 检查是否超出边界
        position = self.data.qpos[0:2]
        if np.any(np.abs(position) > 50):
            return True
        
        return False
    
    def _calculate_drift_angle(self) -> float:
        """计算漂移角度"""
        # 获取车辆前进方向和速度方向的夹角
        forward = np.array([1, 0, 0])  # 车辆前进方向
        velocity = self.data.qvel[0:3]
        
        if np.linalg.norm(velocity) < 0.1:
            return 0.0
        
        # 计算夹角
        velocity_2d = velocity[:2]
        forward_2d = forward[:2]
        
        cos_angle = np.dot(velocity_2d, forward_2d) / (
            np.linalg.norm(velocity_2d) * np.linalg.norm(forward_2d)
        )
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # 确定方向
        cross = np.cross(forward_2d, velocity_2d)
        if cross < 0:
            angle = -angle
        
        return angle
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        # 检查车身是否接触地面
        body_height = self.data.qpos[2]
        if body_height < 0.2:  # 车身高度阈值
            return True
        
        return False 