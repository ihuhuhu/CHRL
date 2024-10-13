import time

import numpy as np
from numba import njit
import Env.Base_env as Base_env
from Env.Base_env import register
import os
import gym
from Env.config import *

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

per = 0.05
per2 = (1 / per - 0.5)
path = os.getcwd()  # 获取当前工作目录路径


@njit(cache=True)
def cal_L2_norm(pos):  # 计算xy到原点的距离
    return np.sqrt((pos ** 2).sum())


class MinicheethMoveEnv(Base_env.Base_env):
    """ minicheeth环境 """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                 GUI=False,
                 seed=None,
                 urdf_root: str = path + '/mini_cheetah/mini_cheetah.urdf',
                 set_time: float = 0,  # 开始前的放置时间
                 start_z: float = 0.48,  # 初始化Z的高度

                 use_ex_f=False,
                 ex_f_co=0.7,
                 max_episode_steps=1200,

                 x_y_random_posture: float = np.pi / 12,  # xy随机姿态
                 use_default_terrain=False,  # 用哪种地形

                 student_frame_stack=100,  # 过去的多少个
                 teacher_frame_stack=4,

                 use_student=False,

                 init_position_high=18,
                 init_position_low=14,
                fallen_time=4,  # 倒下多久结束episode
                 max_distance=19,  # 可运行的最大范围
                 use_random_z=False,  # 全局Z变换
                 use_self_z=True,  # 自身Z变化，走螺旋线

                 heightPerturbationRange=0.03,
                 max_t=33.5,
                 kp=27.948375,
                 ):

        super(MinicheethMoveEnv, self).__init__(
            urdf_root=urdf_root,
            GUI=GUI,
            set_time=set_time,
            x_y_random_posture=x_y_random_posture,
            use_default_terrain=use_default_terrain,
            start_z=start_z,
            use_ex_f=use_ex_f, ex_f_co=ex_f_co, max_episode_steps=max_episode_steps,
            seed=seed,
            student_frame_stack=student_frame_stack,  # 过去的多少个
            teacher_frame_stack=teacher_frame_stack,
            use_student=use_student,
            init_position_high=init_position_high,
            init_position_low=init_position_low, heightPerturbationRange=heightPerturbationRange,
            max_t=max_t, kp=kp,
        )
        self.target_height = 0.3
        self.fallen = 0  # 统计是否倒下，如果倒下超过x秒，则终止
        self.fallen_freq = int(fallen_time * 1 / self.agent_timestep)

        self.max_distance = max_distance

        self.last_action = np.concatenate(
            [(1 - np.array(motor_high) / (np.array(motor_high) - np.array(motor_low))) * 2 - 1,
             np.ones(num_of_motor)])  # 记录上一次的动作

        self.reset_last_message()
        # Z轴是否随机方向
        self.random_euler = np.zeros(3)
        self.use_random_z = use_random_z
        self.target_random_euler = np.concatenate(
            [np.zeros((2,)), self.np_random.uniform(-np.pi / 2.2, np.pi / 2.2, size=(1,))], axis=-1)
        # 自身Z变换
        self.use_self_z = use_self_z

        self.random_pos = None
        self.random_R = None

    def reset_last_message(self):
        self.last_message = {'motor_torque': np.zeros(12),
                             'base_v_world': np.zeros(3),
                             'motor_v': np.zeros(12),
                             # 'origin_base_euler': np.zeros(3),
                             'point_v': np.zeros((8, 3))
                             }

    def update_last_message(self, robot_state):
        for m in self.last_message:
            self.last_message[m] = robot_state[m].copy()

    # def reset_self_z(self, base_p, origin_base_euler):
    #     pass

    def set_command(self, command, random_R):
        self.random_pos = command
        self.random_R = random_R
        if command[0] <= 9:
            self.power_limit = 450
        else:
            self.power_limit = 450 + (command[0] - 9) * 40

    def step(self, action):
        assert self.random_pos is not None
        assert self.random_R is not None
        observation, reward, done, info = self.pd_step(action)
        reward_info, robot_info, student_observation = info
        # robot_info['last_base_p'] = self.last_message['base_p']
        # robot_info['last_origin_base_euler'] = self.last_message['origin_base_euler']

        self.update_last_message(robot_info)  # 更新last euler

        # self.reset_self_z(robot_info['base_p'], robot_info['origin_base_euler'])
        self.random_pos = None
        self.random_R = None
        return observation, reward, done, (reward_info, robot_info, student_observation)

    def reset(self):
        self.power_limit = 450
        # random_action = self.action_space.sample()  # 记录上一次的动作
        self.last_action = np.zeros(action_dim)  # 记录上一次的动作
        self.reset_()
        self.update_last_message(self.robot_state)

        num = 0
        random_action = self.action_space.sample()  # 记录上一次的动作
        if self.use_student:
            step_num = self.np_random.randint(self.student_frame_stack + 5, 115)
        else:
            step_num = self.np_random.randint(self.teacher_frame_stack + 5, 115)
        for _ in range(step_num):  # 连续执行last_action来填充观测
            if self.np_random.binomial(1, 0.1):
                random_action = self.action_space.sample()  # 记录上一次的动作
            self.random_pos = np.array([3, 0])
            self.random_R = np.eye(3)
            observation, reward, done, info = self.step(random_action)
            num += 1
        self.random_pos = None
        self.random_R = None
        # self.next_step_to_rerandom = self.np_random.randint(20, 50)
        self._env_step_counter = 0  # 重置环境步数
        self.fallen = 0
        self.ex_count = 0
        return observation, reward, False, info

    def is_done(self, base_p, v_base_angle, is_fallen):
        if np.sqrt((base_p[0:2] ** 2).sum()) >= self.max_distance:
            return True
        # 倒下计数器
        self.fallen += (is_fallen + (1 - is_fallen) * -1)
        self.fallen = np.clip(self.fallen, 0, np.inf)
        if self._env_step_counter >= self._max_episode_steps or self.fallen >= self.fallen_freq:
            done = True
        else:
            done = False
        return done

    def cal_reward(self, robot_state, is_fallen, v_base_angle):  # 计算奖励
        return 0, {}, {}

    def is_fallen(self, base_p, v_base_angle):  # 是否摔倒
        # raise NotImplementedError
        return float(v_base_angle > 0.45 or
                     base_p[2] < (self.target_height - 0.03) or
                     base_p[2] > (self.target_height + 0.03)
                     )



register(
    id='MinicheethMove-v0',
    entry_point=MinicheethMoveEnv,
)

