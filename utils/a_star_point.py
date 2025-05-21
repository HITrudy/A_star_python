class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf') # 初始化为无穷大
        self.h = 0
        self.parent = None

    # 定义相等性判断
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # 定义哈希支持
    def __hash__(self):
        return hash((self.x, self.y))