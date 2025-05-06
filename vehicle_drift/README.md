# Vehicle Drift Control with Reinforcement Learning

基于强化学习的车辆漂移控制项目，支持Mujoco和Project Chrono两种物理引擎。

## 项目结构

```
vehicle_drift/
├── algorithms/         # 强化学习算法
│   ├── base_algorithm.py  # 算法基础接口
│   └── ppo.py          # PPO算法实现
├── envs/              # 环境实现
│   ├── base_env.py    # 环境基础接口
│   ├── mujoco_env.py  # Mujoco环境实现
│   └── chrono_env.py  # Chrono环境实现
├── utils/             # 工具函数
│   └── trainer.py     # 训练器实现
├── configs/           # 配置文件
│   └── default_config.py  # 默认配置
├── models/            # 模型文件
├── logs/              # 日志文件
├── main.py           # 主程序入口
├── requirements.txt   # 依赖管理
└── README.md         # 项目说明
```

## 安装依赖

```bash
conda activate rl
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 使用Mujoco环境训练
python main.py --env mujoco --algo ppo --mode train

# 使用Chrono环境训练
python main.py --env chrono --algo ppo --mode train

# 使用自定义配置训练
python main.py --env mujoco --algo ppo --mode train --config configs/custom_config.yaml
```

### 评估模型

```bash
# 评估训练好的模型
python main.py --env mujoco --algo ppo --mode eval --model models/best_model.pt
```

## 配置说明

配置文件支持以下主要部分：

- `env`: 环境配置
  - `mujoco`: Mujoco环境参数
  - `chrono`: Chrono环境参数
- `algo`: 算法配置
  - `ppo`: PPO算法参数
- `train`: 训练配置
- `reward`: 奖励函数配置

## 扩展说明

### 添加新的环境

1. 在`envs`目录下创建新的环境类
2. 继承`BaseDriftEnv`并实现所有抽象方法
3. 在`main.py`中添加环境选择逻辑

### 添加新的算法

1. 在`algorithms`目录下创建新的算法类
2. 继承`BaseAlgorithm`并实现所有抽象方法
3. 在`main.py`中添加算法选择逻辑

## 注意事项

1. 使用Mujoco环境需要安装Mujoco物理引擎
2. 使用Chrono环境需要安装Project Chrono
3. 训练过程中会自动保存最佳模型
4. 建议使用GPU进行训练 


1. 训练：python -m vehicle_drift.main --env mujoco --algo ppo --mode train
2. 评测：python -m vehicle_drift.main --env mujoco --algo ppo --mode eval --render --model .\models\best_model.pt --video .\videos\eval