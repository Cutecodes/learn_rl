import gym
from gym import spaces
import numpy as np

class BernoulliBanditEnv(gym.Env):
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        super(BernoulliBanditEnv, self).__init__()
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.action_space = spaces.Discrete(K)  # 动作空间:离散的K个动作,分别对应拉动K根拉杆
        self.observation_space = spaces.Discrete(1)  # 观察空间:单一状态
        self.K = K
        
        self.state = {
            "probs": self.probs,
            "best_idx": self.best_idx,
            "best_prob": self.best_prob
        }
        self.steps = 0
    
    def reset(self):
        """重置环境状态"""
        self.state = {
            "probs": self.probs,
            "best_idx": self.best_idx,
            "best_prob": self.best_prob
        }
        self.steps = 0
        return self.state

    def render(self, mode='human'):
        """渲染环境（可选）"""
        print(f"State: {self.state}")
    
    def close(self):
        """清理资源（可选）"""
        pass       

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        

    
    def step(self, k):
        """执行一步动作"""
        self.steps += 1
        
        # 计算奖励
        if np.random.rand() < self.probs[k]:
            reward = 1
        else:
            reward = 0
        
        # 检查是否终止
        done = False
        
        # 可选：额外信息
        info = {"steps": self.steps}
       
        return self.state, reward, done, info