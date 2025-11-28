from env.mab import BernoulliBanditEnv
import numpy as np
from mab_solver import Solver, plot_results

class DecayingEpsilonGreedySolver(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedySolver, self).__init__(bandit)
        self.total_count = 0.
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.K)

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        _, r, _, _ = self.mab_env.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(0)
env = BernoulliBanditEnv(K=10)  # 创建一个10臂老虎机环境
solver = DecayingEpsilonGreedySolver(env)
solver.run(5000)  

plot_results([solver], ["Decaying Epsilon-Greedy"])