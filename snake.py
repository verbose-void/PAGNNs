import numpy as np
from time import sleep
import os

APPLE_VALUE = 2 
SNAKE_VALUE = 1
VALID_DIRECTIONS = ('U', 'D', 'R', 'L')
deltas = {
    'U': np.array([0, 1]),
    'D': np.array([0, -1]),
    'R': np.array([1, 0]),
    'L': np.array([-1, 0])
}

class Board:
    
    def __init__(self, world_size=(10, 10), choose_direction_function=None, reward_callback=None):
        np.set_printoptions(threshold=20000)
        self.arr = np.zeros(world_size) 
        self.apples_eaten = 0

        # spawn snake
        x = np.random.randint(world_size[0])
        y = np.random.randint(world_size[1])
        self.arr[x, y] = SNAKE_VALUE

        # spawn first apple
        self.spawn_apple()
        self.frame = 0

        self.choose_direction_function = choose_direction_function 
        self.reward_callback = reward_callback


    def get_snake_pos(self):
        x, y = np.where(self.arr == SNAKE_VALUE)
        return x[0], y[0]
    

    def get_apple_pos(self):
        x, y = np.where(self.arr == APPLE_VALUE)
        return x[0], y[0]

    

    def move_snake(self, direction=None):
        # choose direction
        if self.choose_direction_function is not None:
            direction = self.choose_direction_function(self)
        elif direction is None:
            direction = np.random.choice(VALID_DIRECTIONS)

        # validate direction 
        if not direction in VALID_DIRECTIONS:
            raise Exception('Direction %s not found. Valid: %s' % (direction, str(VALID_DIRECTIONS)))

        # handle movement
        x, y = self.get_snake_pos()
        delta = deltas[direction]

        nx, ny = x+delta[0], y+delta[1] 
        if nx < 0 or ny < 0 or nx >= self.arr.shape[0] or ny >= self.arr.shape[1]:
            return

        self.arr[x, y] = 0
        
        # check for "collisions"
        result = None
        if self.arr[nx, ny] == APPLE_VALUE:
            result = 'eat'

        self.arr[nx, ny] = SNAKE_VALUE
        return result
    

    def spawn_apple(self, x=None, y=None):
        if x is None:
            x = np.random.randint(self.arr.shape[0])
        if y is None:
            y = np.random.randint(self.arr.shape[1])

        if self.arr[x, y] != 0:
            return self.spawn_apple()

        self.arr[x, y] = APPLE_VALUE


    def draw(self, suffix=''):
        print(self.arr)
        print('Frame: %i Points: %i %s' % (self.frame, self.apples_eaten, suffix))


    def get_observation(self):
        # the observation is the sensors acting agents have access to
        # for this case, we'll use the delta x & y components for the distance between the agent & the apple 
        sx, sy = self.get_snake_pos()
        ax, ay = self.get_apple_pos()

        return ax - sx, ay - sy 

    
    def reward(self):
        self.apples_eaten += 1
        if self.reward_callback is not None:
            self.reward_callback(self)


    def rollout(self, steps, sleep_length=1, draw=True, suffix=''):
        if draw:
            os.system('clear') 
            self.draw(suffix=suffix)

        for _ in range(steps):
            result = self.move_snake()

            if result is 'eat':
                self.spawn_apple()
                self.reward()
            
            if draw:
                sleep(sleep_length)
                os.system('clear')
                self.draw(suffix=suffix)

            self.frame += 1

if __name__ == '__main__':
    board = Board(world_size=(2, 2))
    board.rollout(60)
    print('Snake ate %i apples!' % board.apples_eaten)
