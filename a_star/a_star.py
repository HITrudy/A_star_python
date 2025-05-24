import time
import numpy as np
import heapq
import sys
sys.path.append("..")
from utils import a_star_point as point
from obs_map.obs_map import ObstacleMap
import matplotlib.pyplot as plt
import random

class AStar:#加入起点终点
    def __init__(self, map, start, end):
        self.map = map
        self.start = start
        self.end = end
        self.open_set = []  # 优先队列（最小堆）
        self.close_set = set()  # 已访问节点集合
        self.counter = 0  # 用于打破优先级相同时的平局
        self.open_dict = {}  # 跟踪开放列表中的节点，用于检测重复

    def HeuristicCost(self, p):#欧式距离，公式等价于求对角线的长度，终点被定义为右上角，传入终点坐标，修改计算。
        x_dis = self.end.x - p.x
        y_dis = self.end.y - p.y
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0 or x >= self.map.shape[0] or y >= self.map.shape[1]:
            return False
        return (self.map[x, y] == 0)

    def IsStartPoint(self, p):
        return p.x == self.start.x and p.y == self.start.y

    def IsEndPoint(self, p):
        return p.x == self.end.x and p.y == self.end.y

    def RunAStar(self):
        # self.SaveInitialState(ax, plt)
        start_point = self.start
        start_point.g = 0
        start_point.h = self.HeuristicCost(start_point)
        heapq.heappush(self.open_set, (start_point.g + start_point.h, self.counter, start_point))
        self.counter += 1
        self.open_dict[(start_point.x, start_point.y)] = start_point
        start_time = time.time()
        while self.open_set:
            _, _, current = heapq.heappop(self.open_set)
            # key = (current.x, current.y)
            # if key in self.open_dict:  
            #     del self.open_dict[key]
            # else:
            #     continue

            if self.IsEndPoint(current):
                return self.BuildPath(current, start_time)

            self.close_set.add((current.x, current.y))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = current.x + dx, current.y + dy
                    self.ProcessPoint(x, y, current, dx, dy)

        print('No path found, algorithm failed!!!')

    def BuildPath(self, p, start_time):
        path = []
        while p is not None:
            path.insert(0, p)
            self.map[p.x, p.y] = 0.5
            if self.IsStartPoint(p):
                break
            p = p.parent
        #构造路径，如果path[0]不是起点，说明构造错误。
        # 如果终点的父节点被更新了，那证明一定有路径？那为什么后面还判断是从起点开始？
        # 防御性测试，在多线程或者实际的工程中，不能保证如果路径没被找到，build_path不会不被调用。
        if not path or not self.IsStartPoint(path[0]):
            print("错误：无法构建完整路径，路径可能不连通！")
            return []
        
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

def random_test():
    map = ObstacleMap(500, mode = 'random', random_para=[20,20])
    while True:
        start = point.Point(random.randint(0, map.size - 1), random.randint(0, map.size - 1))
        end = point.Point(random.randint(0, map.size - 1), random.randint(0, map.size - 1))
        if not (map.map[start.x, start.y] or map.map[end.x, end.y]) : break
    astar = AStar(map.map, start, end)
    astar.RunAStar()
    plt.imshow(astar.map, cmap='Greys', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    random_test()