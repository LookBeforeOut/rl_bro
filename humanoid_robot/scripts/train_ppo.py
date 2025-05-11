import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
from datetime import datetime
from humanoid_robot.src.envs.humanoid_env import HumanoidEnv

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.start_time = datetime.now()
        
    def _on_step(self):
        # 记录每个episode的奖励和长度
        if len(self.model.ep_info_buffer) > 0:
            ep_reward = self.model.ep_info_buffer[-1]["r"]
            ep_length = self.model.ep_info_buffer[-1]["l"]
            self.rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.timesteps.append(self.num_timesteps)
            
            # 每100步打印一次训练信息
            if self.num_timesteps % 100 == 0:
                elapsed_time = datetime.now() - self.start_time
                print(f"\n训练进度:")
                print(f"总步数: {self.num_timesteps}")
                print(f"总episodes: {len(self.rewards)}")
                print(f"最近10个episodes平均奖励: {np.mean(self.rewards[-10:]):.2f}")
                print(f"最近10个episodes平均长度: {np.mean(self.episode_lengths[-10:]):.2f}")
                print(f"训练用时: {elapsed_time}")
                
                # 绘制实时训练曲线
                self._plot_training_curves()
        
        return True
    
    def _plot_training_curves(self):
        if len(self.rewards) < 2:  # 至少需要2个数据点才能绘制
            return
            
        plt.figure(figsize=(12, 4))
        
        # 绘制奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.timesteps, self.rewards, label='Episode Reward')
        if len(self.rewards) >= 10:  # 只在有足够数据点时绘制移动平均
            ma_rewards = self._moving_average(self.rewards, 10)
            ma_timesteps = self.timesteps[-len(ma_rewards):]  # 使用对应的时间步
            plt.plot(ma_timesteps, ma_rewards, label='Moving Average (10)')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        
        # 绘制episode长度曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.timesteps, self.episode_lengths, label='Episode Length')
        if len(self.episode_lengths) >= 10:  # 只在有足够数据点时绘制移动平均
            ma_lengths = self._moving_average(self.episode_lengths, 10)
            ma_timesteps = self.timesteps[-len(ma_lengths):]  # 使用对应的时间步
            plt.plot(ma_timesteps, ma_lengths, label='Moving Average (10)')
        plt.xlabel('Timesteps')
        plt.ylabel('Length')
        plt.title('Episode Lengths')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
    
    def _moving_average(self, data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

def make_env(rank, seed=0, render_mode=None):
    def _init():
        env = HumanoidEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    # 检查GPU是否可用
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU设备名称:", torch.cuda.get_device_name(0))
        print("GPU数量:", torch.cuda.device_count())
        print("当前GPU内存使用:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = "training_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建多个环境实例以并行训练
    n_envs = 4  # 并行环境数量
    # 所有环境使用相同的渲染模式
    env = SubprocVecEnv([make_env(i, seed, render_mode=None) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(0, seed, render_mode=None)])  # 暂时关闭渲染
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    # 创建模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        device="cpu",  # 使用CPU进行训练
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    
    print("模型设备:", next(model.policy.parameters()).device)
    
    # 设置回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="humanoid_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    monitor_callback = TrainingMonitorCallback()
    
    # 开始训练
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, monitor_callback]
    )
    
    # 保存最终模型
    model.save(os.path.join(save_dir, "final_model"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("./logs/best_model", exist_ok=True)
    os.makedirs("./logs/results", exist_ok=True)
    os.makedirs("./logs/tensorboard", exist_ok=True)
    
    main() 