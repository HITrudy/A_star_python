import threading
import queue
import time
import numpy as np
import sys
sys.path.append("..")
from utils import a_star_point as point
from obs_map.obs_map import ObstacleMap
import matplotlib.pyplot as plt
import random
from a_star.a_star import AStar

class ParallelAStar:
    def __init__(self, map, start, end, num_threads=4):
        self.map = map
        self.start = start
        self.end = end
        self.num_threads = num_threads

        # 线程共享数据结构
        self.open_queue = queue.PriorityQueue()  # 线程安全优先队列
        self.close_set = set()
        self.open_dict = {}  # 快速查询节点状态
        self.lock = threading.Lock()  # 共享资源锁
        self.found_event = threading.Event()  # 路径找到事件
        self.counter = 0  # 节点插入顺序计数器
        self.path = None  # 最终路径存储

        # 初始化起点
        start_node = point.Point(start.x, start.y)
        start_node.g = 0
        start_node.h = self.HeuristicCost(start_node)
        self.open_queue.put((start_node.g + start_node.h, self.counter, start_node))
        self.open_dict[(start_node.x, start_node.y)] = start_node
        self.counter += 1

    def HeuristicCost(self, p):
        x_dis = self.end.x - p.x
        y_dis = self.end.y - p.y
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0 or x >= self.map.shape[0] or y >= self.map.shape[1]:
            return False
        return self.map[x, y] == 0

    def IsEndPoint(self, p):
        return p.x == self.end.x and p.y == self.end.y

    def WorkerThread(self):
        while not self.found_event.is_set():
            try:
                # 非阻塞获取节点
                priority, _, current = self.open_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)  # 防止CPU空转
                continue

            # 终点检查
            if self.IsEndPoint(current):
                with self.lock:
                    self.path = self.BuildPath(current)
                    self.found_event.set()
                return

            # 节点处理
            with self.lock:
                if (current.x, current.y) in self.close_set:
                    continue
                self.close_set.add((current.x, current.y))

            # 扩展邻居节点
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    self.ProcessNeighbor(current, dx, dy)

    def ProcessNeighbor(self, parent, dx, dy):
        x = parent.x + dx
        y = parent.y + dy

        # 有效性检查
        if not self.IsValidPoint(x, y):
            return

        # 对角线移动检查相邻节点
        if dx != 0 and dy != 0:
            if not (self.IsValidPoint(parent.x + dx, parent.y) and
                    self.IsValidPoint(parent.x, parent.y + dy)):
                return

        # 计算移动代价
        move_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
        tentative_g = parent.g + move_cost

        # 创建新节点
        neighbor = point.Point(x, y)
        neighbor.h = self.HeuristicCost(neighbor)
        neighbor.parent = parent
        neighbor.g = tentative_g

        with self.lock:
            # 关闭列表检查
            if (x, y) in self.close_set:
                return

            # 开放列表检查
            if (x, y) in self.open_dict:
                existing = self.open_dict[(x, y)]
                if tentative_g < existing.g:
                    existing.g = tentative_g
                    existing.parent = parent
                    self.open_queue.put((existing.g + existing.h, self.counter, existing))
                    self.counter += 1
            else:
                self.open_dict[(x, y)] = neighbor
                self.open_queue.put((neighbor.g + neighbor.h, self.counter, neighbor))
                self.counter += 1

    def BuildPath(self, p):
        path = []
        while p is not None:
            path.insert(0, p)
            self.map[p.x, p.y] = 0.5  # 标记路径
            if p.x == self.start.x and p.y == self.start.y:
                break
            p = p.parent
        return path

    def RunParallel(self):
        threads = []
        start_time = time.time()

        # 创建工作线程
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.WorkerThread)
            threads.append(t)
            t.start()

        # 等待线程完成
        for t in threads:
            t.join()

        # 输出结果
        end_time = time.time()
        if self.path:
            print(f'算法完成，耗时：{end_time - start_time:.2f} 秒')
            print(f'路径长度：{len(self.path)} 步')
        else:
            print("No path found")
        return self.path


# 测试用例
def parallel_test():
    map = ObstacleMap(500, mode='random', random_para=[20, 20])
    while True:
        start = point.Point(random.randint(0, map.size-1), random.randint(0, map.size-1))
        end = point.Point(random.randint(0, map.size-1), random.randint(0, map.size-1))
        if not (map.map[start.x, start.y] or map.map[end.x, end.y]):
            break
    #串行版本
    map1=np.copy(map.map)
    a_star=AStar(map1, start, end)
    a_star.RunAStar()
    # 并行版本
    parallel_astar = ParallelAStar(map.map, start, end, num_threads=4)
    parallel_astar.RunParallel()

    # 可视化
    plt.imshow(parallel_astar.map, cmap='Greys', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    parallel_test()