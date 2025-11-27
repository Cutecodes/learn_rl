from env.mab import BernoulliBanditEnv
import numpy as np
import matplotlib.pyplot as plt

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, env):
        self.K = env.K  # 拉杆个数
        self.mab_env = env
        self.counts = np.zeros(self.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔
    
    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.mab_env.best_prob - self.mab_env.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].K)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = BernoulliBanditEnv(K=10)  # 创建一个10臂老虎机环境
    solver = Solver(env)  # 创建一个多臂老虎机算法求解器
    solver.run(num_steps=1000)  # 运行1000步
