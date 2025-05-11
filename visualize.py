import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_robot.src.envs.humanoid_env import HumanoidEnv
import time

def visualize_model(model_path, vec_normalize_path, num_episodes=5, render_delay=0.01):
    # 创建环境
    env = HumanoidEnv(render_mode="human")  # 启用渲染模式
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False

    # 加载模型
    model = PPO.load(model_path, env=env)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_velocity = 0

        print(f"\nStarting Episode {episode + 1}")
        print("Press 'q' to quit, any other key to continue to next episode...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            episode_velocity += info[0].get('velocity', 0)
            
            # 添加小延迟以便更好地观察
            time.sleep(render_delay)

        print(f"\nEpisode {episode + 1} finished:")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length}")
        print(f"Average Velocity: {episode_velocity/episode_length:.2f} m/s")
        
        # 等待用户输入以继续下一个回合
        if episode < num_episodes - 1:
            user_input = input("\nPress Enter to continue to next episode, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

    env.close()

if __name__ == "__main__":
    # 设置模型和标准化器的路径
    model_path = "training_results/final_model.zip"
    vec_normalize_path = "training_results/vec_normalize.pkl"
    
    # 运行可视化评估
    visualize_model(model_path, vec_normalize_path)