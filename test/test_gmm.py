import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import Slider, ColumnDataSource, RadioButtonGroup, Div
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from sklearn.mixture import GaussianMixture

# -- usage
# bokeh serve --show test_gmm.py

# 1. 生成模拟轨迹数据
def generate_trajectory(start, end, num_points=100, noise=0.1):
    """生成带噪声的轨迹"""
    start = np.array(start)
    end = np.array(end)
    t = np.linspace(0, 1, num_points)
    return start + (end - start) * t[:, np.newaxis] + np.random.normal(0, noise, (num_points, 2))

# 生成3类轨迹数据
np.random.seed(42)
trajectories = []
labels = []
for _ in range(20):  # 直线1
    trajectories.append(generate_trajectory([0, 0], [5, 5]))
    labels.append(0)
for _ in range(20):  # 直线2
    trajectories.append(generate_trajectory([0, 5], [5, 0]))
    labels.append(1)
for _ in range(20):  # 曲线
    t = np.linspace(0, 1, 100)
    traj = np.column_stack([t*5, (t-0.5)**2 * 10 + 1])
    trajectories.append(traj + np.random.normal(0, 0.2, (100, 2)))
    labels.append(2)

X = np.vstack(trajectories)
original_labels = np.repeat(labels, 100)

# 2. 创建数据源
source_original = ColumnDataSource(data={'x': X[:, 0], 'y': X[:, 1], 'label': original_labels})
source_gmm = ColumnDataSource(data={'x': X[:, 0], 'y': X[:, 1], 'label': np.zeros(len(X))})
means_source = ColumnDataSource(data={'x': [], 'y': []})
ellipses_source = ColumnDataSource(data={'xs': [], 'ys': [], 'color': []})

# 3. 创建绘图
tools = "pan,wheel_zoom,box_zoom,reset,save"
plot_original = figure(width=500, height=400, title="真实轨迹类别", tools=tools)
plot_gmm = figure(width=500, height=400, title="GMM聚类结果", tools=tools,
                 x_range=plot_original.x_range, y_range=plot_original.y_range)

# 绘制原始轨迹
colors = Category10[3]
for i in range(3):
    mask = original_labels == i
    plot_original.scatter('x', 'y', source=ColumnDataSource({
        'x': X[mask, 0], 'y': X[mask, 1]}), 
        color=colors[i], alpha=0.3, legend_label=f'Class {i+1}')

# 绘制GMM结果
plot_gmm.scatter('x', 'y', source=source_gmm, color='color', alpha=0.3, size=5, legend_label='聚类点')
plot_gmm.scatter('x', 'y', source=means_source, size=12, marker='x', color='black', legend_label='中心点')
plot_gmm.patches('xs', 'ys', source=ellipses_source, fill_alpha=0.1, line_width=2, legend_label='协方差椭圆')

plot_original.legend.location = "top_right"
plot_gmm.legend.location = "top_right"

# 4. 交互控件
n_components_slider = Slider(title="高斯分布数量", value=3, start=1, end=5, step=1)
cov_type_radio = RadioButtonGroup(labels=["full", "tied", "diag", "spherical"], active=0)
reg_covar_slider = Slider(title="正则化系数 (log10)", value=-6, start=-10, end=-2, step=1)

# 5. 更新函数
def update_gmm():
    n_components = n_components_slider.value
    cov_type = ["full", "tied", "diag", "spherical"][cov_type_radio.active]
    reg_covar = 10 ** reg_covar_slider.value
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        reg_covar=reg_covar,
        random_state=42
    )
    gmm.fit(X)
    pred_labels = gmm.predict(X)
    
    source_gmm.data = {
        'x': X[:, 0],
        'y': X[:, 1],
        'label': pred_labels,
        'color': [Category10[10][i % 10] for i in pred_labels]
    }
    
    means_source.data = {'x': gmm.means_[:, 0], 'y': gmm.means_[:, 1]}
    
    ellipses_data = {'xs': [], 'ys': [], 'color': []}
    for i in range(n_components):
        # 处理不同协方差类型
        if cov_type == 'full':
            cov = gmm.covariances_[i]
        elif cov_type == 'tied':
            cov = gmm.covariances_
        elif cov_type == 'diag':
            cov = np.diag(gmm.covariances_[i])
        else:  # spherical
            cov = np.eye(2) * gmm.covariances_[i]
        
        # 计算椭圆(或圆形)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ
        
        theta = np.linspace(0, 2*np.pi, 100)
        xy = np.column_stack([width/2 * np.cos(theta), height/2 * np.sin(theta)])
        rot = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                       [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
        xy_rot = xy.dot(rot) + gmm.means_[i]
        
        ellipses_data['xs'].append(xy_rot[:, 0])
        ellipses_data['ys'].append(xy_rot[:, 1])
        ellipses_data['color'].append(Category10[10][i % 10])
    
    ellipses_source.data = ellipses_data

update_gmm()

# 6. 回调绑定
for widget in [n_components_slider, reg_covar_slider]:
    widget.on_change('value', lambda attr, old, new: update_gmm())
cov_type_radio.on_change('active', lambda attr, old, new: update_gmm())

# 7. 布局 (不使用Panel/Tabs)
controls = column(
    Div(text="<h2>GMM参数控制</h2>"),
    n_components_slider,
    cov_type_radio,
    reg_covar_slider,
    width=300
)

layout = row(
    column(plot_original, plot_gmm),
    controls
)

curdoc().add_root(layout)
curdoc().title = "GMM轨迹建模(无Panel版)"