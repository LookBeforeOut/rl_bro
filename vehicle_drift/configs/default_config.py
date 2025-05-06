"""默认配置文件"""

# 环境配置
ENV_CONFIG = {
    'mujoco': {
        'model_path': 'models/car.xml',
        'control_freq': 50,
        'sim_freq': 500,
        'render_mode': 'human'
    },
    'chrono': {
        'model_path': 'models/vehicle.chrono',
        'control_freq': 50,
        'sim_freq': 500
    }
}

# 算法配置
ALGO_CONFIG = {
    'ppo': {
        'obs_dim': 13,  # 位置(3) + 姿态(4) + 速度(3) + 角速度(3)
        'act_dim': 2,   # [steering, throttle]
        'hidden_dim': 64,
        'lr': 3e-4,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'target_kl': 0.01,
        'epochs': 10
    }
}

# 训练配置
TRAIN_CONFIG = {
    'max_episodes': 1000,
    'max_steps': 1000,
    'batch_size': 64,
    'save_path': './models/best_model.pt',
    'log_interval': 10,
    'eval_interval': 100,
    'eval_episodes': 10
}

# 奖励配置
REWARD_CONFIG = {
    'drift_angle_weight': 1.0,
    'velocity_weight': 0.1,
    'control_penalty': 0.01,
    'crash_penalty': -10.0
} 