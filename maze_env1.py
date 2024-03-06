import numpy as np
import gym
from gym import spaces
import pygame
import time

class MazeEnv(gym.Env):
    def __init__(self, maze:np.ndarray,x=0,y=0):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space_shape = maze.shape
        low = np.zeros(maze.shape,dtype=np.int64)
        high = np.ones(self.observation_space_shape,dtype=np.int64)
        high = high*5
        #self.observation_space = spaces.Box(low = low,high=high,dtype=np.int64)
        self.observation_space = spaces.MultiDiscrete(np.array([len(maze), len(maze[0])]))
        self.agent_position = self.find_start_position()
        self.goal = self.find_goal_position()

    def find_start_position(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == 1:
                    return (i, j)
        return None

    def find_goal_position(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == 2:
                    return (i, j)
        return None

    def reset(self):
        self.agent_position = self.find_start_position()
        return self.agent_position

    def step(self, action):
        x, y = self.agent_position
        if action == 0:  # Up
            x -= 1
        elif action == 1:  # Down
            x += 1
        elif action == 2:  # Left
            y -= 1
        elif action == 3:  # Right
            y += 1

        reward = -5  # Default reward for each step

        if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]):
            if self.maze[x][y] != 0:
                self.agent_position = (x, y)
                if self.maze[x][y] == 2:  # Reached the goal
                    reward = 100
                    done = True
                else:
                    done = False
            else:
                done = False
                reward = -5
        else:
            done = False

        l = self.maze.copy()
        l[self.agent_position[0],self.agent_position[1]] = 2
        l[self.goal[0],self.goal[1]] = 4
        if(self.goal[0]==self.agent_position[0] and self.agent_position[1] == self.goal[1]):
            l[self.goal[0], self.goal[1]] = 5
        return self.agent_position, reward, done, {}

    def render(self,*args, **kwargs):
        cell_width = 30
        cell_height = 30
        screen_width = len(self.maze[0]) * cell_width
        screen_height = len(self.maze) * cell_height

        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        screen.fill((255, 255, 255))

        # Draw maze
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == 0:  # Wall
                    pygame.draw.rect(screen, (0, 0, 0), (j * cell_width, i * cell_height, cell_width, cell_height))
                elif self.maze[i][j] == 2:  # Goal
                    pygame.draw.rect(screen, (0, 255, 0), (j * cell_width, i * cell_height, cell_width, cell_height))

        # Draw agent
        agent_x, agent_y = self.agent_position
        pygame.draw.circle(screen, (255, 0, 0), (agent_y * cell_width + cell_width // 2, agent_x * cell_height + cell_height // 2), min(cell_width, cell_height) // 3)

        pygame.display.flip()

        time.sleep(0.1)

maze = np.array([
    [1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 2]
])

if __name__ == '__main__':
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

    env = MazeEnv(maze)
    #env.render()
    #time.sleep(4)

    #Example usage:
    while True:
        env.render()
        action = env.action_space.sample()  # Replace with your RL model's action selection
        observation, reward, done, _ = env.step(action)
        print("action : ",action," Obs : ",observation," reward : ",reward," done : ",done)
        #time.sleep(1)
        if done:
            print("Goal reached!")
            break