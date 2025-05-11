import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_robot.src.envs.humanoid_env import HumanoidEnv

def evaluate_model(model_path, vec_normalize_path, num_episodes=10):
    # 创建环境
    env = HumanoidEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # 评估时不需要训练
    env.norm_reward = False  # 评估时不需要标准化奖励

    # 加载模型
    model = PPO.load(model_path, env=env)

    # 评估指标
    episode_rewards = []
    episode_lengths = []
    episode_velocities = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_velocity = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            episode_velocity += info[0].get('velocity', 0)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_velocities.append(episode_velocity / episode_length)

        print(f"Episode {episode + 1}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length}")
        print(f"Average Velocity: {episode_velocity/episode_length:.2f} m/s")
        print("-" * 50)

    # 打印总体统计信息
    print("\nOverall Statistics:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Velocity: {np.mean(episode_velocities):.2f} ± {np.std(episode_velocities):.2f} m/s")

if __name__ == "__main__":
    # 设置模型和标准化器的路径
    model_path = "models/humanoid_ppo"
    vec_normalize_path = "models/vec_normalize.pkl"
    
    # 运行评估
    evaluate_model(model_path, vec_normalize_path) 