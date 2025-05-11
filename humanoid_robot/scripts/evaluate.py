import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from humanoid_robot.src.envs.humanoid_env import HumanoidEnv

def main():
    # 创建环境
    env = DummyVecEnv([lambda: HumanoidEnv()])
    env = VecNormalize.load("./logs/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    
    # 加载模型
    model = PPO.load("./logs/best_model/best_model")
    
    # 运行评估
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main() 