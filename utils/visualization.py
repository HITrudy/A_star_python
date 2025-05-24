import matplotlib.pyplot as plt
import numpy as np

def draw_circle(c_x, c_y, c_r):
    theta = np.linspace(0, 2 * np.pi, 100)  # 参数化角度
    x = c_x + c_r * np.cos(theta)  # 圆的x坐标
    y = c_y + c_r * np.sin(theta)  # 圆的y坐标
    plt.plot(x, y, 'r-', linewidth=2)  # 绘制圆

def draw_rectangle(s_x, s_y, s_width, s_height):

    # 矩形的四个顶点
    x1, y1 = s_x, s_y
    x2, y2 = s_x + s_width, s_y
    x3, y3 = s_x + s_width, s_y + s_height
    x4, y4 = s_x, s_y + s_height

    # 绘制矩形的四条边
    plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)  # 底边
    plt.plot([x2, x3], [y2, y3], 'b-', linewidth=2)  # 右边
    plt.plot([x3, x4], [y3, y4], 'b-', linewidth=2)  # 顶边
    plt.plot([x4, x1], [y4, y1], 'b-', linewidth=2)  # 左边

def visualize_obstacles(map, path):
    """
    可视化圆形障碍物、矩形障碍物和路径。

    参数:
    circle_obstacles: [[c_x, c_y, c_r], ...]，圆形障碍物的中心坐标和半径
    square_obstacles: [[s_x, s_y, s_width, s_height], ...]，矩形障碍物的左下角坐标、宽度和高度
    path: [(x1, y1), (x2, y2), ...]，路径点的坐标列表
    """
    circle_obstacles = map.circle_obstacles
    square_obstacles = map.square_obstacles


    for c_x, c_y, c_r in circle_obstacles:
        draw_circle(c_x, c_y, c_r)
    
    for s_x, s_y, s_width, s_height in square_obstacles:
        draw_rectangle(s_x, s_y, s_width, s_height)


    path_x = []
    path_y = []
    for point in path:
        path_x.append(point.x)
        path_y.append(point.y)

    plt.plot(path_x, path_y, 'g--', linewidth=2, label='Path')  # 绿色虚线表示路径
    plt.legend()
    plt.axis('equal')
    plt.show()
