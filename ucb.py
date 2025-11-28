from env.mab import BernoulliBanditEnv
import numpy as np
from mab_solver import Solver, plot_results

class UCBSolver(Solver):
    """ 上置信界算法,继承Solver类 """
    def __init__(self, bandit, coef=1, init_prob=1.0):
        super(UCBSolver, self).__init__(bandit)
        self.coef = coef
        self.total_count = 0.
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.K)

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        _, r, _, _ = self.mab_env.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(0)
env = BernoulliBanditEnv(K=10)  # 创建一个10臂老虎机环境
solver = UCBSolver(env)
solver.run(5000)  

plot_results([solver], ["UCB"])