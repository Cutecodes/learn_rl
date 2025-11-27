from env.mab import BernoulliBanditEnv
import numpy as np
from mab_solver import Solver, plot_results

class EpsilonGreedySolver(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedySolver, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        _, r, _, _ = self.mab_env.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(2)
env = BernoulliBanditEnv(K=10)  # 创建一个10臂老虎机环境
solver = EpsilonGreedySolver(env)  # 创建一个多臂老虎机算法求解器
solver.run(num_steps=5000)  # 运行1000步

print("Epsilon-Greedy算法累积懊悔:", solver.regret)
plot_results([solver], ["EpsilonGreedy"])