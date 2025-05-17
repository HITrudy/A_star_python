import matplotlib.pyplot as plt
import numpy as np
import random 

class ObstacleMap:
    def __init__(self, size, 
                 circle_obstacles = None, 
                 square_obstacles = None,
                 mode = 'default',
                 random_para = None):
        '''
        circle_obstacles = [[c_x,c_y,c_r], [c_x,c_y,c_r], ...] center of circle and radius of circle
        square_obstacles = [[s_x,s_y,s_width,s_height], [s_x,s_y,s_width,s_height]] bottom left corner of square , width and height of square
        random_para = [circle_random_num, square_random_num]
        '''
        self.map = np.zeros((size, size))
        self.size = size

        self.circle_obstacles = circle_obstacles
        self.square_obstacles = square_obstacles
        self.random_para = random_para
        self.random_radius = 10
        
        if mode == 'random':
            self.generate_random_map()
            return 
        
        if mode == 'default':
            self.update_map_by_circle()
            self.update_map_by_square()


    def update_map_by_circle(self):
        for x in range(len(self.map)):
            for y in range(len(self.map)):
                for obs in self.circle_obstacles:
                    distance = np.sqrt((x - obs[0]) ** 2 + (y - obs[1]) ** 2)
                    if distance <= obs[2]:
                        self.map[x, y] = 1
                        break
    
    def update_map_by_square(self):
        for obs in self.square_obstacles:
            obs = np.floor(obs).astype(int)
            for x in range(obs[2]):
                for y in range(obs[3]):
                    if obs[0]+x >= self.size or obs[1]+y >= self.size: break
                    self.map[obs[0]+x, obs[1]+y] = 1

    def generate_random_map(self):
        self.circle_obstacles = []
        for _ in range(self.random_para[0]):
            self.circle_obstacles.append([random.random() * self.size, random.random() * self.size, random.random() * self.random_radius])
        self.update_map_by_circle()

        self.square_obstacles = []
        for _ in range(self.random_para[1]):
            self.square_obstacles.append([random.random() * self.size, random.random() * self.size, random.random() * self.random_radius, random.random() * self.random_radius])
        self.update_map_by_square()


if __name__ == "__main__" :
    map = ObstacleMap(100, mode = 'random', random_para=[10,10])
    plt.imshow(map.map, cmap='Greys', interpolation='nearest')
    plt.show()
# print([random.random() * 50, random.random() * 50, random.random() * 3])
# print(np.random.randint(0, 50, 2).append(np.random.randint(1,3,1)))