import time
import numpy as np
import heapq
import sys
sys.path.append("..")
from utils import ha_star_point as point
from utils import visualization
from obs_map.obs_map import ObstacleMap
import matplotlib.pyplot as plt
import random
from math import cos, sin, tan, floor
import dubins

class HAStar:#加入起点终点
    def __init__(self, map, start, end):
        self.map = map
        self.start = start
        self.end = end
        self.open_set = []  
        self.close_set = set()  
        self.counter = 0  
        self.open_dict = {}  
        self.step_size = 2
        self.wheelbase = 3

    def HeuristicCost(self, p):
        start = (p.x, p.y, p.phi)
        end = (self.end.x, self.end.y, self.end.phi)
        return dubins.shortest_path(start, end, self.wheelbase/sin(0.6)).path_length()
    
    # # Euclidean distance
    # def HeuristicCost(self, p):
    #     x_dis = self.end.x - 1 - p.x
    #     y_dis = self.end.y - 1 - p.y
    #     return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0 or x >= self.map.shape[0] or y >= self.map.shape[1]:
            return False
        return (self.map[x, y] == 0)

    def IsStartPoint(self, p):
        return floor(p.x) == floor(self.start.x) and floor(p.y) == floor(self.start.y) and floor(p.phi*10) == floor(self.start.phi*10)

    def IsEndPoint(self, p):
        return floor(p.x) == floor(self.end.x) and floor(p.y) == floor(self.end.y) and floor(p.phi*10) == floor(self.end.phi*10)

    def RunHAStar(self):
        start_point = self.start
        start_point.g = 0
        start_point.h = self.HeuristicCost(start_point)
        heapq.heappush(self.open_set, (start_point.g + start_point.h, self.counter, start_point))
        self.counter += 1
        self.open_dict[(floor(start_point.x), floor(start_point.y), floor(start_point.phi*10))] = start_point
        start_time = time.time()
        while self.open_set:
            _, _, current = heapq.heappop(self.open_set)

            # print(current.x,current.y)
            if self.IsEndPoint(current):
                return self.BuildPath(current, start_time)
            # self.map[floor(current.x), floor(current.y)] = 0.5
            self.close_set.add((floor(current.x), floor(current.y), floor(current.phi*10)))

            for delta in [-0.6, -0.6/2, 0, 0.6/2, 0.6]:
                self.ProcessPoint(current, delta)
        print('No path found, algorithm failed!!!')
        return []

    def BuildPath(self, p, start_time):
        path = []
        while p is not None:
            path.insert(0, p)
            self.map[floor(p.x), floor(p.y)] = 0.5
            if self.IsStartPoint(p):
                break
            p = p.parent
        if not path or not self.IsStartPoint(path[0]):
            print("错误：无法构建完整路径，路径可能不连通！")
            return []
        
        end_time = time.time()
        print(f'算法完成，耗时：{end_time - start_time:.2f} 秒')
        print(f'路径长度：{len(path)} 步')
        return path


    def ProcessPoint(self, current, delta):
        x = current.x + self.step_size*cos(current.phi)
        y = current.y + self.step_size*sin(current.phi)
        phi = (current.phi + self.step_size*tan(delta)/self.wheelbase) % 6.28

        if not self.IsValidPoint(floor(x), floor(y)):
            return
        if (floor(x), floor(y), floor(phi*10)) in self.close_set:
            return
        
        neighbor = point.Point(x, y, phi)

        move_cost = self.step_size
        delta_cost = abs(current.delta - delta) * 1.0
        g = current.g + move_cost + delta_cost

        if (floor(x), floor(y), floor(phi*10)) in self.open_dict:
            existing_node = self.open_dict[(floor(x), floor(y), floor(phi*10))]
            if g < existing_node.g:
                existing_node.parent = current  # 确保更新父节点
                existing_node.g = g
                heapq.heappush(self.open_set, (g + existing_node.h, self.counter, existing_node))
                self.counter += 1
            return
        else:
            neighbor.h = self.HeuristicCost(neighbor)
            neighbor.parent = current
            neighbor.g = g
            heapq.heappush(self.open_set, (neighbor.g + neighbor.h, self.counter, neighbor))
            self.counter += 1
            self.open_dict[(floor(x), floor(y), floor(phi*10))] = neighbor

def random_test():
    map = ObstacleMap(100, mode = 'random', random_para=[10,0])
    # while True:
    #     start = point.Point(random.randint(0,99), random.randint(0,99))
    #     end = point.Point(random.randint(0,99), random.randint(0,99))
    #     if not (map.map[start.x, start.y] or map.map[end.x, end.y]) : break
    start = point.Point(0, 0, 0)
    end = point.Point(90, 90, 0)
    astar = HAStar(map.map, start, end)
    path = astar.RunHAStar()
    if len(path) == 0 : return
    visualization.visualize_obstacles(map, path)
    # plt.imshow(astar.map, cmap='Greys', interpolation='nearest')
    # plt.show()

if __name__ == "__main__":
    for _ in range(12):
        random_test()