import time
import numpy as np
from matplotlib.patches import Rectangle
import heapq
import point
#open_dict的作用，
class AStar:#加入起点终点
    def __init__(self, map, start, end):
        self.map = map
        self.start = start
        self.end = end
        self.open_set = []  # 优先队列（最小堆）
        self.close_set = set()  # 已访问节点集合
        self.counter = 0  # 用于打破优先级相同时的平局
        self.open_dict = {}  # 跟踪开放列表中的节点，用于检测重复
#这个map根据传入到参数的决定的，在main函数中，我们传入了random_map，故这个map会指向我们的ranodnmmap

    def HeuristicCost(self, p):#欧式距离，公式等价于求对角线的长度，终点被定义为右上角，传入终点坐标，修改计算。
        x_dis = self.end.x - 1 - p.x
        y_dis = self.end.y - 1 - p.y
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0 or x >= self.map.size or y >= self.map.size:
            return False
        return not self.map.IsObstacle(x, y)

# 起点，终点也不是对点。
    def IsStartPoint(self, p):
        return p.x == self.start.x and p.y == self.start.y

    def IsEndPoint(self, p):
        return p.x == self.end.x and p.y == self.end.y

    def RunAndSaveImage(self, ax, plt):
        #总结来说：这个方法干了三件事，一，初始化，将起点加入最小堆和字典中，二，取出对应堆顶节点，三，遍历8个方向的节点，
        self.SaveInitialState(ax, plt)
        start_point = self.start
        start_point.g = 0
        start_point.h = self.HeuristicCost(start_point)
        #将open_set加入到最小堆中，评价指标是括号内的内容，这一步还是在初始化，了解一下最小堆怎么实现的
        heapq.heappush(self.open_set, (start_point.g + start_point.h, self.counter, start_point))
        self.counter += 1
        #open_dict的作用，可以快速查找open_set中的内容
        self.open_dict[(start_point.x, start_point.y)] = start_point
        start_time = time.time()
        #因为最小堆中存着同一个节点的不同实例，故我们会先判断是否在open_dict（一定是最新的点）中
        while self.open_set:
            #取出堆顶节点，按三元组存储的，取的时候我们也是对应位置取出
            _, _, current = heapq.heappop(self.open_set)
            key = (current.x, current.y)
            if key in self.open_dict:  # 检查键是否存在
                del self.open_dict[key]
            else:
                continue  # 跳过已移除的旧条目

            if self.IsEndPoint(current):
                return self.BuildPath(current, ax, plt, start_time)

            self.close_set.add((current.x, current.y))
            #处理8个方向的节点
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = current.x + dx, current.y + dy
                    self.ProcessPoint(x, y, current, dx, dy)

        print('No path found, algorithm failed!!!')

    def BuildPath(self, p, ax, plt, start_time):
        path = []

        while p is not None:
            path.insert(0, p)
            if self.IsStartPoint(p):
                break
            p = p.parent
        #构造路径，如果path[0]不是起点，说明构造错误。
        # 如果终点的父节点被更新了，那证明一定有路径？那为什么后面还判断是从起点开始？
        # 防御性测试，在多线程或者实际的工程中，不能保证如果路径没被找到，build_path不会不被调用。
        if not path or not self.IsStartPoint(path[0]):
            print("错误：无法构建完整路径，路径可能不连通！")
            return []
        self.SaveFinalPath(path, ax, plt)
        end_time = time.time()
        print(f'算法完成，耗时：{end_time - start_time:.2f} 秒')
        print(f'路径长度：{len(path)} 步')
        return path


    def ProcessPoint(self, x, y, parent, dx, dy):
        if not self.IsValidPoint(x, y):
            return
        if (x, y) in self.close_set:
            return
        #如果不是上下左右这4个方向则，则移动代价为根号二。
        move_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
        neighbor = point.Point(x, y)
        #当前要加入的邻居节点的g值计算。
        tentative_g_cost = parent.g + move_cost
        neighbor.h = self.HeuristicCost(neighbor)
        #如果这邻居节点之前已经加入到字典中，则进行更新操作。为什么用字典不要set,
        if (x, y) in self.open_dict:
            existing_node = self.open_dict[(x, y)]
            #如果这个邻居节点的g值比之前小，则更新，加入到最小堆中，如果比之前还大，则不理睬。
            if tentative_g_cost < existing_node.g:
                existing_node.parent = parent  # 确保更新父节点
                existing_node.g = tentative_g_cost
                heapq.heappush(self.open_set, (tentative_g_cost + existing_node.h, self.counter, existing_node))
                self.counter += 1
            return
        else:
            #如果没有则加入
            neighbor.parent = parent  # 确保设置父节点
            neighbor.g = tentative_g_cost
            #更新最小堆。
            heapq.heappush(self.open_set, (neighbor.g + neighbor.h, self.counter, neighbor))
            self.counter += 1
            self.open_dict[(x, y)] = neighbor


    def SaveInitialState(self, ax, plt):
        plt.imshow(self.map.map, cmap='Greys', interpolation='nearest')
        ax.add_patch(Rectangle((self.start.x,self.start.y), 1, 1, color='red', label='起点'))
        ax.add_patch(Rectangle((self.end.x, self.end.y), 1, 1, color='blue', label='终点'))
        plt.title('初始地图（红色=起点，蓝色=终点，黑色=障碍物）')
        plt.legend()
        plt.draw()
        plt.savefig('initial_state.png', dpi=300)

    def SaveFinalPath(self, path, ax, plt):
        for p in path:
            ax.add_patch(Rectangle((p.x, p.y), 1, 1, color='green', alpha=0.5))
        plt.title('寻路结果（绿色=路径）')
        plt.savefig('final_path.png', dpi=300)
