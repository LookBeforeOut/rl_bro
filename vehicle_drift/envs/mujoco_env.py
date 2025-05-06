import os
os.environ['MUJOCO_GL'] = 'glfw'  # 强制使用 GLFW 后端

import mujoco
import numpy as np
from typing import Dict, Tuple, Any, Optional
import cv2
import matplotlib.pyplot as plt
from vehicle_drift.envs.base_env import BaseDriftEnv
import time
from collections import deque

class MujocoDriftEnv(BaseDriftEnv):
    """优化后的Mujoco车辆漂移环境，增强可视化功能"""
    
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
        
        # 渲染相关设置
        self.render_mode = config.get('render_mode', None)
        self.viewer = None
        self._init_viewer()
        
        # 视频录制相关
        self.video_writer = None
        self.video_path = None
        self.frame_queue = deque(maxlen=100)  # 用于存储渲染帧
        
        # 轨迹记录
        self.trajectory = []
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # 控制参数
        self.control_freq = config.get('control_freq', 50)
        self.sim_freq = config.get('sim_freq', 500)
        self.sim_steps = int(self.sim_freq / self.control_freq)
        
        # 性能统计
        self.render_times = []
        self.step_times = []
        
    def _init_viewer(self):
        """初始化viewer，支持多种渲染模式"""
        if self.render_mode is None:
            return
            
        try:
            if self.render_mode == 'human':
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.azimuth = 180  # 调整相机角度
                self.viewer.cam.elevation = -20
                self.viewer.cam.distance = 10.0
                
            elif self.render_mode == 'rgb_array':
                # 用于离屏渲染
                self.viewer = mujoco.Renderer(self.model)
                
        except Exception as e:
            print(f"Warning: Failed to initialize viewer: {e}")
            print("Continuing without rendering...")
            self.viewer = None
    
    def start_video_recording(self, video_path: str, fps: int = 30):
        """开始录制视频，支持更高效的帧处理
        
        Args:
            video_path: 视频保存路径
            fps: 视频帧率
        """
        self.video_path = video_path
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # 根据渲染模式确定分辨率
        if self.render_mode == 'human':
            resolution = (1280, 720)  # 假设viewer窗口大小
        else:
            resolution = (800, 600)
        
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            resolution
        )
        
        # 清空帧队列
        self.frame_queue.clear()
    
    def stop_video_recording(self):
        """停止录制视频并保存"""
        if self.video_writer is not None:
            # 写入队列中剩余的帧
            while self.frame_queue:
                self.video_writer.write(self.frame_queue.popleft())
                
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved to {self.video_path}")
    
    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        start_time = time.time()
        
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始位置和速度
        self.data.qpos[0:3] = [0, 0, 0.5]  # 位置
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 姿态（四元数）
        self.data.qvel[0:3] = [0, 0, 0]  # 线速度
        self.data.qvel[3:6] = [0, 0, 0]  # 角速度
        
        # 重置轨迹和奖励记录
        self.trajectory = []
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        # 如果是被动viewer，同步初始状态
        if self.viewer is not None and hasattr(self.viewer, 'sync'):
            self.viewer.sync()
        
        # 记录重置时间
        self.step_times.append(time.time() - start_time)
        
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作，优化性能统计"""
        step_start = time.time()
        
        # 应用动作
        self.data.ctrl[0] = action[0] * 0.5  # 转向 [-0.5, 0.5]
        self.data.ctrl[1] = action[1] * 100  # 油门 [0, 100]
        
        # 前向仿真
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
        
        # 记录轨迹
        self.trajectory.append({
            'position': self.data.qpos[0:3].copy(),
            'orientation': self.data.qpos[3:7].copy(),
            'velocity': self.data.qvel[0:3].copy(),
            'angular_velocity': self.data.qvel[3:6].copy(),
            'action': action.copy(),
            'timestamp': time.time()
        })
        
        # 获取观察值
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._calculate_reward()
        self.current_episode_reward += reward
        
        # 检查是否结束
        done = self._check_done()
        
        # 额外信息
        info = {
            'velocity': np.linalg.norm(self.data.qvel[0:3]),
            'drift_angle': self._calculate_drift_angle(),
            'position': self.data.qpos[0:3].copy(),
            'orientation': self.data.qpos[3:7].copy(),
            'episode': {
                'r': self.current_episode_reward,
                'l': len(self.trajectory)
            }
        }
        
        # 渲染
        if self.render_mode is not None:
            render_start = time.time()
            self.render()
            self.render_times.append(time.time() - render_start)
        
        # 记录步骤时间
        self.step_times.append(time.time() - step_start)
        
        return obs, reward, done, info
    
    def render(self, mode: Optional[str] = None):
        """优化的渲染方法，支持多种模式"""
        if mode is None:
            mode = self.render_mode
            
        if mode is None:
            return
            
        if mode == 'human' and self.viewer is not None:
            try:
                self.viewer.sync()
                
                # 获取当前帧用于视频录制
                if self.video_writer is not None:
                    # 从viewer获取图像
                    if hasattr(self.viewer, 'read_pixels'):
                        frame = self.viewer.read_pixels()
                        if frame is not None:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            self.frame_queue.append(frame_bgr)
                            if len(self.frame_queue) >= 5:  # 批量写入提高效率
                                while self.frame_queue:
                                    self.video_writer.write(self.frame_queue.popleft())
                    
            except Exception as e:
                print(f"Warning: Failed to render: {e}")
                self.viewer = None
                
        elif mode == 'rgb_array' and self.viewer is not None:
            try:
                self.viewer.update_scene(self.data)
                frame = self.viewer.render()
                if self.video_writer is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(frame_bgr)
                return frame
            except Exception as e:
                print(f"Warning: Failed to render rgb_array: {e}")
                return None
    
    def close(self):
        """关闭环境并保存性能数据"""
        if self.viewer is not None:
            if hasattr(self.viewer, 'close'):
                self.viewer.close()
            self.viewer = None
            
        if self.video_writer is not None:
            self.stop_video_recording()
        
        # 保存性能统计
        if self.step_times:
            avg_step_time = np.mean(self.step_times)
            print(f"Average step time: {avg_step_time*1000:.2f}ms")
            
        if self.render_times:
            avg_render_time = np.mean(self.render_times)
            print(f"Average render time: {avg_render_time*1000:.2f}ms")
    
    def _get_obs(self) -> np.ndarray:
        """获取观察值"""
        return np.concatenate([
            self.data.qpos[0:3],    # 位置 (3)
            self.data.qpos[3:7],    # 姿态 (4)
            self.data.qvel[0:3],    # 速度 (3)
            self.data.qvel[3:6]     # 角速度 (3)
        ]).astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """优化的奖励计算"""
        velocity = np.linalg.norm(self.data.qvel[0:3])
        drift_angle = self._calculate_drift_angle()
        
        # 主奖励项
        reward = (
            1.0 * abs(drift_angle) +  # 漂移角度权重
            0.1 * velocity            # 速度权重
        )
        
        # 控制平滑度惩罚
        if len(self.trajectory) > 1:
            last_action = self.trajectory[-2]['action']
            action_diff = np.linalg.norm(last_action - self.data.ctrl[:2])
            reward -= 0.02 * action_diff  # 动作变化惩罚
        
        # 碰撞惩罚
        if self._check_collision():
            reward -= 10.0
            
        return reward
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        return self._check_collision() or np.any(np.abs(self.data.qpos[0:2]) > 50)
    
    def _calculate_drift_angle(self) -> float:
        """计算漂移角度，优化数值稳定性"""
        velocity = self.data.qvel[0:3]
        if np.linalg.norm(velocity) < 0.1:
            return 0.0
            
        forward = np.array([1, 0])  # 车辆前进方向(2D)
        velocity_2d = velocity[:2]
        
        # 使用arctan2提高数值稳定性
        angle = np.arctan2(velocity_2d[1], velocity_2d[0]) - np.arctan2(forward[1], forward[0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-π, π]
        
        return angle
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        return self.data.qpos[2] < 0.2  # 车身高度阈值
    
    def save_trajectory_plot(self, save_path: str, show_actions: bool = True):
        """增强的轨迹可视化，支持动作显示"""
        if not self.trajectory:
            return
            
        positions = np.array([t['position'] for t in self.trajectory])
        x, y = positions[:, 0], positions[:, 1]
        
        plt.figure(figsize=(12, 12))
        
        # 绘制轨迹
        plt.plot(x, y, 'b-', alpha=0.7, label='Trajectory')
        plt.scatter(x[0], y[0], color='g', s=100, label='Start')
        plt.scatter(x[-1], y[-1], color='r', s=100, label='End')
        
        # 添加动作信息
        if show_actions and len(self.trajectory) > 1:
            actions = np.array([t['action'] for t in self.trajectory])
            steering = actions[:, 0]
            throttle = actions[:, 1]
            
            # 每隔10步显示一次动作
            for i in range(0, len(x), 10):
                if i < len(x):
                    dx = 0.5 * steering[i]
                    dy = 0.5 * throttle[i]
                    plt.arrow(x[i], y[i], dx, dy, 
                             head_width=0.2, head_length=0.2, fc='purple', ec='purple', alpha=0.5)
        
        # 添加轨迹方向箭头
        for i in range(0, len(x), 20):
            if i + 1 < len(x):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                plt.arrow(x[i], y[i], dx, dy, 
                         head_width=0.3, head_length=0.3, fc='k', ec='k', alpha=0.7)
        
        plt.title(f'Vehicle Trajectory (Total Reward: {self.current_episode_reward:.2f})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_episode_report(self, save_dir: str, episode_num: int):
        """保存完整的回合报告，包括轨迹、奖励和统计数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存轨迹图
        traj_path = os.path.join(save_dir, f'episode_{episode_num}_trajectory.png')
        self.save_trajectory_plot(traj_path)
        
        # 保存奖励曲线
        if len(self.episode_rewards) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(self.episode_rewards, 'b-o')
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'reward_history.png'))
            plt.close()