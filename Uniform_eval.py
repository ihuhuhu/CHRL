import time
import torch
import numpy as np
import random
from Env import MinicheethMoveEnv
import gym
import os
from utils.Agent import Agent
from Main.Command import Uniform_Command

urdf_root = os.getcwd() + '/Env/mini_cheetah/mini_cheetah.urdf'


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 6

set_seed_everywhere(seed)

# ==========环境===========
env = gym.make('MinicheethMove-v0', urdf_root=urdf_root, GUI=True, seed=seed, use_default_terrain=False,
               use_ex_f=False)  # 构造环境
# init_Random_coefficient['foot_Friction_low'] = 1.0
# env.Env_random.Random_coefficient = init_Random_coefficient

env.action_space.seed(seed)  # 设置随机动作采样的种子
state_dim = env.teacher_observation_space.shape[0]  # state的维度
action_dim = env.action_space.shape[0]  # action的维度

# ==========agent==========
learned_agent = Agent(state_dim, action_dim, )
file_name = './eval/'
learned_agent.load(file_name)


# learned_agent.min_uncertain = -0.13
# learned_agent.update_dist()


# ========设定评估=========
def update_eval_uncertain_list(agent, eval_uncertain_num=12):
    uncertain_list = np.arange(0, eval_uncertain_num) / (eval_uncertain_num - 1)
    uncertain_list = list(uncertain_list * (agent.uncertain - agent.min_uncertain) + agent.min_uncertain)
    return uncertain_list


eval_uncertain_list = [0.3]

command = Uniform_Command(4)

eval_freq = 500


def print_mean_std(n, mean, std, around=3):
    print(n, ':', np.around(mean, around), u'\u00B1', np.around(std, around))


for uncertain in eval_uncertain_list:

    for x in range(100):
        state, reward, done, info = env.reset()

        video_name = str(uncertain) + '_' + str(x) + '.mp4'
        reward_info = None
        all_r = 0

        step_ = 0
        last_time = time.time()
        while 1:
            # Recorder.record(env)
            step_ += 1
            state = command.change_state(state, None, env.use_self_z)
            env.set_command(command.random_pos, command.random_R[-1, :, :])

            action = learned_agent.select_best_action(state, uncertain=uncertain)

            state, reward, done, info = env.step(action)
            print(state.size())

            base_p = info[1]['base_p']

            res_time = time.time() - last_time
            # print(res_time, env.this_time)
            if res_time < env.this_time:
                time.sleep(env.this_time - res_time)
            last_time = time.time()

            if abs(base_p[0]) > 90 or abs(base_p[1]) > 90:
                env.reset()

env.close()
