import gym
from gym import spaces
import numpy as np

class CliffWalkingEnv(gym.Env):
    """ 悬崖漫步环境 """
    def __init__(self, ncol, nrow):
        super(CliffWalkingEnv, self).__init__()
        self.ncol = ncol
        self.nrow = nrow
        self.steps = 0
        self.x = 0
        self.y = self.nrow - 1
    
    def reset(self):
        """重置环境状态"""
        self.x = 0
        self.y = self.nrow - 1
        self.steps = 0
        return self.y * self.ncol + self.x

    def render(self, mode='human'):
        """渲染环境（可选）"""
        print(f"State: {self.state}")
    
    def close(self):
        """清理资源（可选）"""
        pass       

    def step(self, action):
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        self.steps += 1
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        info = {"steps": self.steps}
        return next_state, reward, done, info
