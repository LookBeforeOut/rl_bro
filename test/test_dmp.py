import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import Slider, ColumnDataSource
from bokeh.layouts import column, row
from bokeh.io import curdoc
import argparse

# # 创建解析器
# parser = argparse.ArgumentParser(description="代码演示用法")

# # 添加参数
# parser.add_argument('-r', '--run', type=str, help='用法')
# parser.add_argument('-a', '--age', type=int, help='你的年龄')

# args = parser.parse_args()

# bokeh serve --show test_dmp.py

# 1. DMP核心实现
class DMP:
    def __init__(self, n_bfs=10, alpha_y=25, beta_y=6.25, alpha_x=1.0):
        self.n_bfs = n_bfs          # 基函数数量
        self.alpha_y = alpha_y      # 弹簧刚度
        self.beta_y = beta_y        # 阻尼系数
        self.alpha_x = alpha_x      # 相位变量衰减率
        
        # 初始化基函数中心和时间宽度
        self.c = np.exp(-alpha_x * np.linspace(0, 1, n_bfs))
        self.h = 1.0 / (np.diff(self.c)**2)
        self.h = np.append(self.h, self.h[-1])
        
        # 权重（通过演示轨迹学习）
        self.w = np.zeros(n_bfs)
    
    def learn_from_demo(self, y_demo, dt, tau=1.0):
        """从演示轨迹学习权重w"""
        n_samples = len(y_demo)
        x = 1.0
        y, dy, ddy = y_demo[0], 0, 0
        
        # 计算目标f
        f_target = []
        x_log = []
        for t in range(1, n_samples):
            # 数值微分计算速度和加速度
            dy = (y_demo[t] - y_demo[t-1]) / (dt * tau)
            if t > 1:
                ddy = (y_demo[t] - 2*y_demo[t-1] + y_demo[t-2]) / (dt * tau)**2
            
            # 目标f
            f = (tau**2 * ddy) - self.alpha_y * (self.beta_y * (y_demo[-1] - y) - tau * dy)
            f_target.append(f)
            
            # 更新相位变量
            x += (-self.alpha_x * x) * dt * tau
            x_log.append(x)
            
            y = y_demo[t]
        
        # 局部加权回归学习权重w
        psi = np.exp(-self.h[:, None] * (np.array(x_log) - self.c[:, None])**2)
        self.w = np.sum(psi * np.array(f_target), axis=1) / np.sum(psi, axis=1)
        
        print(f"self.w: {self.w}")
    
    def generate_trajectory(self, y0, g, dt, tau=1.0, n_steps=100):
        """生成轨迹"""
        y, dy = y0, 0
        x = 1.0
        
        Y = [y0]
        T = [0]
        
        for t in range(1, n_steps):
            # 更新相位变量x
            x += (-self.alpha_x * x) * dt * tau
            
            # 计算非线性扰动项f
            psi = np.exp(-self.h * (x - self.c)**2)
            f = np.sum(psi * self.w) * x * (g - y0) / np.sum(psi)
            
            # 更新加速度、速度、位置
            ddy = (self.alpha_y * (self.beta_y * (g - y) - tau * dy) + f) / tau**2
            dy += ddy * dt
            y += dy * dt
            
            Y.append(y)
            T.append(t * dt)
        
        return np.array(T), np.array(Y)

# 2. 生成演示轨迹（示例：一条正弦波）
dt = 0.01
t_demo = np.arange(0, 1, dt)
y_demo = np.sin(t_demo * np.pi) * 0.5 + 0.5

# 3. 初始化DMP并学习
dmp = DMP(n_bfs=10)
dmp.learn_from_demo(y_demo, dt)

# 4. Bokeh交互界面
source_demo = ColumnDataSource(data={'t': t_demo, 'y': y_demo})
source_dmp = ColumnDataSource(data={'t': [], 'y': []})
source_dmp_pt = ColumnDataSource(data={'goal_x': [], 'goal_y': []})

plot = figure(width=600, height=400, title="DMP轨迹生成")
plot.line('t', 'y', source=source_demo, line_width=2, color="blue", legend_label="演示轨迹")
plot.line('t', 'y', source=source_dmp, line_width=2, color="red", legend_label="DMP轨迹")
plot.scatter(x=[0], y=[y_demo[0]], size=10, color="green", legend_label="起点")
plot.scatter(x=[1], y=[y_demo[-1]], size=10, color="orange", legend_label="原终点")
plot.scatter('goal_x', 'goal_y', source=source_dmp_pt, size=10, color="red", legend_label="新终点")
plot.legend.location = "top_left"

# 滑块控件
slider_g = Slider(title="新目标位置 (g)", value=y_demo[-1], start=-1.0, end=2.0, step=0.1)
slider_tau = Slider(title="时间缩放 (tau)", value=1.0, start=0.5, end=2.0, step=0.1)
slider_bfs = Slider(title="基函数数量", value=10, start=5, end=20, step=1)


def update(attr, old, new):
    # 更新DMP参数
    dmp.n_bfs = int(slider_bfs.value)
    dmp.c = np.exp(-dmp.alpha_x * np.linspace(0, 1, dmp.n_bfs))
    dmp.h = 1.0 / (np.diff(dmp.c)**2)
    dmp.h = np.append(dmp.h, dmp.h[-1])
    dmp.w = np.zeros(dmp.n_bfs)
    dmp.learn_from_demo(y_demo, dt)
    
    # 生成新轨迹
    t, y = dmp.generate_trajectory(
        y0=y_demo[0], 
        g=slider_g.value, 
        dt=dt, 
        tau=slider_tau.value
    )

    # 更新绘图数据
    source_dmp.data = {
        't': t,
        'y': y,
    }
    source_dmp_pt.data = {
        'goal_x': [t[-1]],
        'goal_y': [y[-1]]
    }


slider_g.on_change('value', update)
slider_tau.on_change('value', update)
slider_bfs.on_change('value', update)

# 初始更新
update(None, None, None)

# 布局
inputs = column(slider_g, slider_tau, slider_bfs)
layout = row(plot, inputs)

curdoc().add_root(layout)