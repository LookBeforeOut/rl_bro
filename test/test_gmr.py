import numpy as np
from sklearn.mixture import GaussianMixture
from bokeh.plotting import figure, curdoc
from bokeh.models import Slider, ColumnDataSource, Button
from bokeh.layouts import column, row
from bokeh.palettes import Category10
import scipy

# 1. 生成带速度的演示数据
def generate_demo_trajectories():
    """生成轨迹数据 [time, x, y, dx, dy]"""
    np.random.seed(42)
    trajectories = []
    
    # 直线轨迹
    for _ in range(5):
        t = np.linspace(0, 1, 50)
        x = t * 2
        y = t * 1
        dx = np.full_like(t, 2.0)
        dy = np.full_like(t, 1.0)
        trajectories.append(np.column_stack([t, x, y, dx, dy]))
    
    # 曲线轨迹
    for _ in range(5):
        t = np.linspace(0, 1, 50)
        x = t * 2
        y = np.sin(t * np.pi) * 0.5 + 0.5
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        trajectories.append(np.column_stack([t, x, y, dx, dy]))
    
    return np.vstack(trajectories)

# 2. 稳健的GMR实现
class GMR:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components, 
                                 covariance_type='full',
                                 reg_covar=1e-6)  # 防止奇异矩阵
        self.n_components = n_components
    
    def fit(self, X):
        self.gmm.fit(X)
        
    def predict(self, t_target):
        """返回: [x, y, dx, dy]"""
        # 添加维度确保输入是2D
        t_target = np.array([[t_target, 0, 0, 0, 0]])  
        responsibilities = self.gmm.predict_proba(t_target)[0]
        
        pred = np.zeros(4)  # x, y, dx, dy
        
        for k in range(self.n_components):
            mu_k = self.gmm.means_[k]
            sigma_k = self.gmm.covariances_[k]
            
            # 分割时间和其他变量
            mu_t = mu_k[0]
            mu_xy = mu_k[1:3]
            mu_vel = mu_k[3:]
            
            sigma_tt = sigma_k[0, 0]
            sigma_txy = sigma_k[0, 1:3]
            sigma_tvel = sigma_k[0, 3:]
            
            # 条件均值计算 (简化稳定版)
            delta_t = t_target[0, 0] - mu_t
            cond_xy = mu_xy + (sigma_txy / (sigma_tt + 1e-6)) * delta_t
            cond_vel = mu_vel + (sigma_tvel / (sigma_tt + 1e-6)) * delta_t
            
            pred[:2] += responsibilities[k] * cond_xy
            pred[2:] += responsibilities[k] * cond_vel
        
        return pred

# 3. 准备数据
demo_data = generate_demo_trajectories()
gmr = GMR(n_components=3)
gmr.fit(demo_data)

# 4. Bokeh可视化设置
source_demo = ColumnDataSource(data={
    'x': demo_data[:, 1], 
    'y': demo_data[:, 2]
})

source_pred = ColumnDataSource(data={
    'x': [], 'y': [],
    'vx': [], 'vy': [],
    'x_end': [], 'y_end': []  # 速度箭头终点
})

# 创建绘图
plot = figure(width=700, height=500, 
              title="GMR轨迹规划 (红色:轨迹, 蓝色:速度方向)",
              tools="pan,wheel_zoom,box_zoom,reset")

# 演示数据
plot.scatter('x', 'y', source=source_demo, 
            color="gray", alpha=0.2, size=3,
            legend_label="演示数据")

# 预测轨迹
plot.line('x', 'y', source=source_pred,
         line_width=3, color="red",
         legend_label="预测轨迹")

# 速度箭头 (使用segment而不是ray)
plot.segment(x0='x', y0='y', x1='x_end', y1='y_end',
            source=source_pred, color="blue",
            line_width=1, alpha=0.7,
            legend_label="速度方向")

# 交互控件
slider_time = Slider(title="目标时间", value=0.5, 
                    start=0, end=1, step=0.01)
button_generate = Button(label="生成轨迹", 
                        button_type="success")

def update_trajectory():
    t_points = np.linspace(0, slider_time.value, 30)
    results = np.array([gmr.predict(t) for t in t_points])
    
    # 计算速度箭头终点 (缩放因子0.1)
    arrow_scale = 0.1
    x_end = results[:, 0] + results[:, 2] * arrow_scale
    y_end = results[:, 1] + results[:, 3] * arrow_scale
    
    source_pred.data = {
        'x': results[:, 0],
        'y': results[:, 1],
        'vx': results[:, 2],
        'vy': results[:, 3],
        'x_end': x_end,
        'y_end': y_end
    }

button_generate.on_click(update_trajectory)

# 初始状态
update_trajectory()

# 图例和布局
plot.legend.location = "top_left"
plot.legend.click_policy = "hide"  # 点击图例可隐藏元素

layout = column(
    plot,
    row(slider_time, button_generate)
)

curdoc().add_root(layout)
curdoc().title = "机器人GMR轨迹规划(速度场修正版)"