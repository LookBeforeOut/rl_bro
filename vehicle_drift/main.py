import argparse
import os
from typing import Dict, Any
import yaml

from vehicle_drift.envs.mujoco_env import MujocoDriftEnv
# from vehicle_drift.envs.chrono_env import ChronoDriftEnv
from vehicle_drift.algorithms.ppo import PPO
from vehicle_drift.utils.trainer import Trainer
from vehicle_drift.configs.default_config import ENV_CONFIG, ALGO_CONFIG, TRAIN_CONFIG, REWARD_CONFIG

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    if config_path is None or not os.path.exists(config_path):
        return {
            'env': ENV_CONFIG,
            'algo': ALGO_CONFIG,
            'train': TRAIN_CONFIG,
            'reward': REWARD_CONFIG
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Vehicle Drift Control with RL')
    parser.add_argument('--env', type=str, default='mujoco', choices=['mujoco', 'chrono'],
                        help='Environment to use (mujoco or chrono)')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo'],
                        help='RL algorithm to use')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode to run (train or eval)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file for evaluation')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建环境
    if args.env == 'mujoco':
        env = MujocoDriftEnv(config['env']['mujoco'])
    # else:
    #     env = ChronoDriftEnv(config['env']['chrono'])
    
    # 创建算法
    if args.algo == 'ppo':
        algo = PPO(config['algo']['ppo'])
    
    # 创建训练器
    trainer = Trainer(env, algo, config['train'])
    
    if args.mode == 'train':
        # 训练模型
        trainer.train(config['train']['save_path'])
    else:
        # 评估模型
        if args.model:
            algo.load(args.model)
        metrics = trainer.evaluate()
        print("Evaluation Metrics:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"Std Reward: {metrics['std_reward']:.2f}")
        print(f"Min Reward: {metrics['min_reward']:.2f}")
        print(f"Max Reward: {metrics['max_reward']:.2f}")

if __name__ == '__main__':
    main() 