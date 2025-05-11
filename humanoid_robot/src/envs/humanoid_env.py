import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from mujoco.glfw import glfw
import os
import time
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle
import mujoco.viewer

class HumanoidEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 40,  # 修改为40fps以匹配dt
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.viewer = None
        
        # 获取模型文件路径
        model_path = os.path.join(os.path.dirname(__file__), "../../config/humanoid.xml")
        
        # 初始化MujocoEnv
        super().__init__(
            model_path=model_path,
            frame_skip=5,
            observation_space=None,  # 将在_setup_spaces中设置
            default_camera_config={},
            render_mode=render_mode
        )
        
        # 设置动作空间和观察空间
        self._setup_spaces()
        
        EzPickle.__init__(self)
        
        # 稳定性检查参数
        self.max_angular_velocity = 10.0  # 最大角速度
        self.max_linear_velocity = 5.0    # 最大线速度
        self.max_joint_acceleration = 20.0 # 最大关节加速度
        self.unstable_count = 0          # 不稳定计数器
        self.max_unstable_steps = 10     # 最大允许不稳定步数

    def _setup_spaces(self):
        # 设置动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),  # 17个关节
            dtype=np.float32
        )
        
        # 设置观察空间
        obs_dim = 70  # 使用与训练时相同的维度
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def step(self, action):
        # 执行动作
        self.do_simulation(action, self.frame_skip)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._get_reward()
        
        # 检查是否结束
        terminated = self._is_done()
        truncated = False
        
        # 获取额外信息
        info = self._get_info()
        
        # 如果需要渲染
        if self.render_mode == "human":
            self._render_frame()
        
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def reset_model(self):
        # 重置模型状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始位置（稍微抬高一点，避免直接接触地面）
        self.data.qpos[2] = 1.25  # 设置初始高度
        
        # 重置不稳定计数器
        self.unstable_count = 0
        
        # 返回初始观察
        return self._get_obs()

    def _get_obs(self):
        # 获取当前状态作为观察
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        
        # 组合位置和速度信息，并填充到70维
        obs = np.zeros(70, dtype=np.float32)
        obs[:len(qpos)] = qpos
        obs[len(qpos):len(qpos)+len(qvel)] = qvel
        
        return obs

    def _get_reward(self):
        # 计算奖励
        # 这里使用一个简单的奖励函数，你可以根据需要修改
        velocity = self.data.qvel[0]  # 前进速度
        alive_bonus = 1.0  # 存活奖励
        return velocity + alive_bonus

    def _is_done(self):
        # 检查是否结束
        # 这里可以添加更多的终止条件
        return False

    def _get_info(self):
        # 返回额外信息
        return {
            "velocity": self.data.qvel[0],
            "position": self.data.qpos[0]
        }

    def _render_frame(self):
        """渲染当前帧"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    model=self.model,
                    data=self.data
                )
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _check_stability(self):
        # 检查角速度
        if np.any(np.abs(self.data.qvel[3:]) > self.max_angular_velocity):
            return False
            
        # 检查线速度
        if np.any(np.abs(self.data.qvel[:3]) > self.max_linear_velocity):
            return False
            
        # 检查关节加速度
        if np.any(np.abs(self.data.qacc) > self.max_joint_acceleration):
            return False
            
        # 检查是否摔倒（躯干高度过低）
        if self.data.qpos[2] < 0.5:
            return False
            
        return True

    def _compute_reward(self):
        # 基础奖励：保持直立
        height_reward = self.data.qpos[2] - 1.25  # 相对于初始高度的偏差
        
        # 速度奖励：鼓励向前移动
        velocity_reward = self.data.qpos[0]  # x方向的位置
        
        # 能量惩罚：减少不必要的运动
        energy_penalty = 0.1 * np.sum(np.square(self.data.ctrl))
        
        # 稳定性奖励
        stability_reward = 0.0
        if self._check_stability():
            stability_reward = 1.0
        
        # 总奖励
        reward = height_reward + velocity_reward - energy_penalty + stability_reward
        
        return reward 