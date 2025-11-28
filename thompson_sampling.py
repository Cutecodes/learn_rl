from env.mab import BernoulliBanditEnv
import numpy as np
from mab_solver import Solver, plot_results

class ThompsonSamplingSolver(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSamplingSolver, self).__init__(bandit)
        self._a = np.ones(self.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆

        _, r, _, _ = self.mab_env.step(k)  # 得到本次动作的奖励
        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k

np.random.seed(0)
env = BernoulliBanditEnv(K=10)  # 创建一个10臂老虎机环境
solver = ThompsonSamplingSolver(env)
solver.run(5000)  

plot_results([solver], ["Thompson Sampling"])