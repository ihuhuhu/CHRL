import copy
import random
import time
from copy import deepcopy
import numpy as np
from gym import spaces
from Env.MinicheethRobot import *
from gym.envs.registration import registry
import os
from Env.functions import random_f_xyz
from scipy.spatial.transform import Rotation
from Env.utlis import desp_limit

RENDER_HEIGHT = 1280
RENDER_WIDTH = 1920

path = os.getcwd()  # 获取当前工作目录路径
rng = np.random.default_rng()


def cal_acc(robot_state, g, pd_timestep):  # 计算加速度计数据
    # imu_v, imu_State = self.get_imu_v(origin_R)
    base_a = (robot_state['imu_v'] - robot_state['last_imu_v']) / pd_timestep
    g = robot_state['origin_imu_R'].reshape(3, 3).T @ g.reshape(3, 1)
    base_a = - g.reshape(3) + base_a
    base_a /= 9.815
    acc = base_a.clip(-8, 8)  # IMU的最大量程
    return acc


class Base_env(Minicheeth):
    """ base minicheeth环境 """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                 urdf_root: str = path + '/mini_cheetah/mini_cheetah.urdf',
                 # 获取默认储存路径
                 GUI=False,  # 是否渲染
                 use_randomize=True,  # 是否对环境随机绕动
                 use_default_terrain=False,  # 用哪种地形
                 set_time: float = 2,  # 开始前的放置时间
                 x_y_random_posture: float = np.pi / 2,  # xy随机姿态
                 start_z: float = 0.5,  # 初始化Z的高度
                 ex_f_co=1.0,

                 # PD
                 LPF=0.9,  # 一阶低通滤波器

                 use_ex_f=False,  # 突然绕动

                 # =====agent=====
                 agent_timestep=0.015,  # agent的频率66.6HZ
                 agent_timestep_random=0.001,  # agent的频率的波动
                 lag=0.006,  # 延迟
                 state_lag=0.007,  # 延迟
                 seed=None,  # 种子
                 max_episode_steps=1000,
                 student_frame_stack=100,  # 过去的多少个
                 teacher_frame_stack=4,

                 use_student=False,

                 init_position_high=18,
                 init_position_low=14,
                 heightPerturbationRange=0.02,
                 max_t=33.5,

                 kp=27.948375,
                 ):
        super(Base_env, self).__init__(urdf_root=urdf_root,
                                       GUI=GUI,
                                       use_randomize=use_randomize,
                                       use_default_terrain=use_default_terrain,
                                       set_time=set_time,
                                       x_y_random_posture=x_y_random_posture,
                                       start_z=start_z,
                                       init_position_high=init_position_high,
                                       init_position_low=init_position_low,
                                       seed=seed, heightPerturbationRange=heightPerturbationRange, max_t=max_t,
                                       )

        self.agent_timestep = agent_timestep
        self.agent_frq = int(agent_timestep * 1000)
        self.agent_freq_random = int(agent_timestep_random * 1000)  # agent的频率的波动
        self.agent_freq_random = np.arange(-self.agent_freq_random, self.agent_freq_random + 1, 1)

        # self.lag_freq = int(lag * 1000)  # 延迟

        self.student_frame_stack = student_frame_stack
        self.teacher_frame_stack = teacher_frame_stack
        self.use_student = use_student
        self.LPF = LPF

        self.action_space = spaces.Box(-np.ones(shape=action_dim), np.ones(shape=action_dim), dtype=np.float64)

        self.teacher_observation_space = spaces.Box(
            -np.ones(shape=teacher_observation_dim * teacher_frame_stack) * 1e5,
            np.ones(shape=teacher_observation_dim * teacher_frame_stack) * 1e5,
            dtype=np.float64)
        self.teacher_observation_buffer = [np.zeros(teacher_observation_dim) for _ in range(teacher_frame_stack)]

        if use_student:
            self.student_observation_space = spaces.Box(
                -np.ones(shape=student_observation_dim * student_frame_stack) * 1e5,
                np.ones(shape=student_observation_dim * student_frame_stack) * 1e5,
                dtype=np.float64)
            self.student_observation_buffer = [np.zeros(student_observation_dim) for _ in range(student_frame_stack)]

        # self.last_action也是观测的一部分

        self._cam_dist = 1.5  # cam摄像机参数
        self._cam_yaw = 0
        self._cam_pitch = -30

        self._env_step_counter = 0  # 记录环境步数
        self._max_episode_steps = max_episode_steps

        self.last_action = np.zeros(action_dim)  # 记录上一次的动作

        self.feedforward_t_zeros = np.zeros(num_of_motor)
        # self.reset()  # 环境重置
        self.ex_count = 0
        self.use_ex_f = use_ex_f
        if use_ex_f:
            print('使用随机强绕动')
        else:
            print('没用')

        self.ex_f_co = ex_f_co

        self.force_temp = None
        self.this_time = 0.015
        self.kp = kp

        self.des_p = np.zeros(num_of_motor)
        self.motor_delay = motor_delay(max_len=6, size=num_of_motor, np_random=self.np_random)
        self.robot_state_delay = robot_state_delay(max_len=6, np_random=self.np_random)

        self.power_limit = 450
        self.protect_des_p = np.zeros(action_dim)
        self.action_lag = np.array([np.random.choice([0, 1, 2], size=3, replace=False) for _ in range(4)]).reshape(-1)

    def step(self, action):

        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def set_use_ex_f(self, value):
        self.use_ex_f = value

    def set_Random_coefficient(self, Random_coefficient):
        self.Env_random.set_Random_coefficient(Random_coefficient)

    def PD_with_protect(self, des_p, kp, kd, power):
        delay_motor_p, delay_motor_v = self.motor_delay.sample(delay_low=1, delay_high=4)
        # ==================================
        max_p = delay_motor_p + 33 / self.kp
        min_p = delay_motor_p - 33 / self.kp

        p_high, p_low, t_v = desp_limit(des_p, kd, delay_motor_v, delay_motor_p, kp)
        des_p_high_, des_p_low_ = fig_des_p(delay_motor_p)
        des_p = np.clip(des_p,
                        np.maximum.reduce([p_low, des_p_low_, motor_low_1, min_p]),
                        np.minimum.reduce([p_high, des_p_high_, motor_high_1, max_p]))

        des_p, torque, t_v = power_limit(des_p, kd, delay_motor_v, delay_motor_p, kp, t_v=t_v, max_power=power)
        des_p = torque_limit(des_p, kd, delay_motor_v, delay_motor_p, kp, torque=torque, t_v=t_v, target=85)
        des_p = np.clip(des_p,
                        np.maximum.reduce([des_p_low_, motor_low_1]),
                        np.minimum.reduce([des_p_high_, motor_high_1]))
        # =================然后是执行动作延迟==========================

        lag = self.total_it % 3
        action_lag_flag = (self.action_lag == lag).astype(np.float64)
        self.protect_des_p = des_p * action_lag_flag + self.protect_des_p * (1 - action_lag_flag)

        self.PD_control(self.kp, self.protect_des_p, 0, self.feedforward_t_zeros)

    def execute_action(self, des_p):
        # self.PD_control(self.kp, des_p, 0, self.feedforward_t_zeros)
        self.PD_with_protect(des_p, self.kp, self.motor_kd, self.power_limit)
        if self.use_student:
            self.robot_state_delay.append(copy.deepcopy(self.robot_state), self.all_random_message['gravity'])
        self.power_list.append(self.robot_state['power'])
        self.foot_contact_list.append(self.robot_state['foot_contact'])
        self.foot_v_list.append(self.cal_foot_v())
        self.torque_list.append(self.robot_state['true_torque'])
        self.motor_delay.append(self.robot_state['motor_p'], self.robot_state['motor_v'])

    def pd_step(self, action):
        if self.use_student:
            self.update_all = True
        else:
            self.update_all = False

        # # 干扰计数器
        # if self.ex_count == 4:
        #     self.ex_count = 0
        # elif self.ex_count != 0:
        #     self.ex_count += 1

        assert action.shape == (action_dim,)
        student_observation = None
        self.power_list = []
        self.foot_contact_list = []
        self.foot_v_list = []
        self.torque_list = []

        last_des_p = self.des_p.copy()

        # 首先是计算延迟
        cal_lag_freq = self.np_random.randint(3, 6)
        for lag in range(cal_lag_freq):  # 机器人执行
            self.execute_action(last_des_p)

        # ===============然后计算动作======================
        # delay_motor_p, delay_motor_v = self.motor_delay.sample(delay_low=1, delay_high=4)
        self.last_action = self.LPF * self.last_action + (1 - self.LPF) * action  # 一阶低通滤波器
        self.des_p = normalize_action(des_p_low, des_p_high, self.last_action)

        self.des_p = np.clip(self.des_p,
                             np.maximum.reduce([last_des_p - 0.2, motor_low_1]),
                             np.minimum.reduce([last_des_p + 0.2, motor_high_1]))

        self.last_action = reverse_action(des_p_low, des_p_high, self.des_p)

        # max_action_lag = 0
        # last_des_p = self.des_p
        # assert (last_des_p == self.des_p).all()

        '''注意这里kp是常数'''
        lag_freq = cal_lag_freq

        # =========延迟完毕，执行动作========

        # =========控制频率随机===========
        if self.use_randomize:
            agent_frq = self.agent_frq + int(self.np_random.choice(self.agent_freq_random))
        else:
            agent_frq = self.agent_frq

        self.this_time = agent_frq * self._time_step  #

        for x in range(agent_frq - lag_freq - 1):  # 机器人执行，需要减去延迟的指令
            self.execute_action(self.des_p)

        # ==============最后额外执行一次==================
        self.update_all = True
        # self.PD_control(self.kp, last_des_p, 0, self.feedforward_t_zeros)
        self.PD_with_protect(self.des_p, self.kp, self.motor_kd, self.power_limit)

        reward_robot_state = copy.deepcopy(self.robot_state)

        if self.use_ex_f and self.np_random.binomial(1, 1 / 300):

            v = self.np_random.uniform(1.0, 2.5) * self.np_random.choice([-1, 1])
            a = self.np_random.uniform(-np.pi, np.pi)
            b = np.arccos(self.np_random.uniform(-0.5, 0.5))

            random_v = v * np.array([np.sin(a) * np.sin(b), np.cos(a) * np.sin(b), np.cos(b)]) + \
                       self.robot_state['base_v_world']

            w = self.np_random.uniform(1.0, 2.5) * self.np_random.choice([-1, 1])
            a = self.np_random.uniform(-np.pi, np.pi)
            b = np.arccos(self.np_random.uniform(-1, 1))
            axis = np.array([np.sin(a) * np.sin(b),
                             np.cos(a) * np.sin(b),
                             np.cos(b)])
            random_w = Rotation.from_rotvec(axis * w).as_euler('xyz')
            random_w += self.robot_state['base_w_world']
            self._pybullet_client.resetBaseVelocity(self.quadruped, random_v, random_w)

            origin_R_T = self.robot_state['origin_R'].reshape(3, 3).T

            self.robot_state['base_w_world'] = random_w
            self.robot_state['base_v_world'] = random_v
            self.robot_state['base_v'] = (origin_R_T @ random_v.reshape(3, 1)).flatten()  # 机体坐标系的线速度
            self.robot_state['base_w'] = (origin_R_T @ random_w.reshape(3, 1)).flatten()  # 机体坐标系的线速度

            self.robot_state['imu_v'], imu_State, self.robot_state['origin_imu_R'], self.robot_state['imu_w'], \
            self.robot_state['imu_w_world'] = self.get_imu_v()

            self.robot_state['point_v'] = self.get_point_v()

            self.ex_count = 1
        else:
            self.ex_count = 0

        # 执行后刷新随机变量
        if self.ex_count == 0 and self.use_randomize:
            # if self.use_randomize:
            # self.Env_random.update_random_message(self.np_random, self.base_mass, self.leg_mass, LPF=0.975,
            #                                       p=1 / 300)
            self.Env_random.apply(self._pybullet_client, self.quadruped, self.terrain)
            self.all_random_message = self.Env_random.get_all_random()

        if self.use_student:
            self.robot_state_delay.append(copy.deepcopy(self.robot_state), self.all_random_message['gravity'])
        self.power_list.append(self.robot_state['power'])
        self.foot_contact_list.append(self.robot_state['foot_contact'])
        self.foot_v_list.append(self.cal_foot_v())
        self.torque_list.append(self.robot_state['true_torque'])
        self.motor_delay.append(self.robot_state['motor_p'], self.robot_state['motor_v'])
        self.update_all = False

        assert len(self.power_list) == len(self.foot_contact_list) == len(self.foot_v_list) == len(
            self.torque_list) == agent_frq

        # =============计算奖励================
        v_base_angle = cal_base_angle(self.robot_state['origin_R'])  # 与Z
        is_fallen = self.is_fallen(self.robot_state['base_p'], v_base_angle)
        reward, reward_info, other_info = self.cal_reward(reward_robot_state, is_fallen, v_base_angle)
        robot_info = {
            'this_time': self.this_time,
        }
        robot_info = {**robot_info, **self.robot_state, **other_info}

        self._env_step_counter += 1
        teacher_observation = self.get_teacher_observation(self.robot_state)
        if self.use_student:
            # ob也是有延迟
            # robot_state, g = self.robot_state_delay.sample(delay_low=1, delay_high=5)
            delay_motor_p, delay_motor_v = self.motor_delay.sample(delay_low=1, delay_high=4)
            student_observation = self.get_student_observation(self.robot_state['base_p'], delay_motor_p, delay_motor_v)

        return teacher_observation, reward, self.is_done(self.robot_state['base_p'], v_base_angle, is_fallen), (
            reward_info, robot_info, student_observation)

    def get_teacher_observation(self, robot_state):
        base_v, base_p, base_w, base_R, motor_p, motor_v, foot_contact = \
            robot_state['base_v'], robot_state['base_p'], robot_state['base_w'], robot_state['base_R'], \
            robot_state['true_motor_p'], robot_state['true_motor_v'], robot_state['foot_contact']

        observation = np.concatenate(
            [base_p, self.last_action, base_v, base_w, base_R, motor_p, motor_v, foot_contact,
             self.all_random_message['gravity'],
             [self.all_random_message['base_mass'] + self.all_random_message['point_mass'].sum(),
              self.all_random_message['foot_Friction_mean'],
              ],
             ])

        assert observation.shape == (teacher_observation_dim,)
        self.teacher_observation_buffer = self.teacher_observation_buffer[1:]
        self.teacher_observation_buffer.append(observation)
        observation = np.concatenate(self.teacher_observation_buffer)
        assert observation.shape == (teacher_observation_dim * self.teacher_frame_stack,)
        return observation

    def get_student_observation(self, base_p, motor_p, motor_v):
        robot_state, g, i = self.robot_state_delay.sample(delay_low=1, delay_high=5)
        imu_R, imu_w = robot_state['imu_R'], robot_state['imu_w']

        robot_state, g, i = self.robot_state_delay.sample(delay_low=1, delay_high=5)
        acc = cal_acc(robot_state, g, self.pd_timestep)

        observation = np.concatenate([base_p[:2], self.last_action, imu_w, motor_p, motor_v, acc, imu_R])

        assert observation.shape == (student_observation_dim,)
        self.student_observation_buffer = self.student_observation_buffer[1:]
        self.student_observation_buffer.append(observation)

        observation = np.concatenate(self.student_observation_buffer)
        observation = observation.reshape(self.student_frame_stack, -1)
        base_p = base_p[:2][None, :].repeat(self.student_frame_stack, 0)
        observation[:, :2] = base_p

        observation = observation.flatten()
        assert observation.shape == (student_observation_dim * self.student_frame_stack,)
        return observation

    def is_done(self, base_p, v_base_angle, is_fallen):
        raise NotImplementedError

    def render(self, mode="rgb_array"):  # 渲染
        if mode != "rgb_array":
            return np.array([])
        base_pos, base_o = self.robot_state['base_p'], self.robot_state['base_o']
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(width=RENDER_WIDTH,
                                                                height=RENDER_HEIGHT,
                                                                viewMatrix=view_matrix,
                                                                projectionMatrix=proj_matrix,
                                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        # 返回RBG图像
        return rgb_array

    def cal_reward(self, robot_state, is_fallen, v_base_angle):  # 计算奖励
        raise NotImplementedError

    def is_fallen(self, base_p, v_base_angle):  # 是否摔倒
        raise NotImplementedError

    def cal_foot_v(self):
        foot_v = np.linalg.norm(self.robot_state['foot_v'], ord=2, axis=-1)
        return foot_v


def normalize_action(low, high, action):  # 输入(-1, 1)的动作,转为env中原本的动作
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action


def reverse_action(low, high, action):  # 把env中原本的动作,转为(-1, 1)的动作
    action = 2 * (action - low) / (high - low) - 1
    action = np.clip(action, -1, 1)
    return action


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


def cal_base_angle(R):
    z = np.array([0, 0, 1])
    new_v = (z.reshape(1, 3) @ R.reshape(3, 3)).reshape(3)
    cos = np.multiply(z, new_v).sum()
    angle = np.arccos(cos)
    assert angle >= 0
    # print(1, angle)

    return angle


class motor_delay:
    def __init__(self, max_len, size, np_random):
        # ==========延迟=============
        self.delay_motor_p = [np.zeros(size) for _ in range(max_len)]
        self.delay_motor_v = [np.zeros(size) for _ in range(max_len)]
        self.max_len = max_len
        self.i2 = np.arange(0, size)
        self.np_random = np_random

    def append(self, motor_p, motor_v):
        self.delay_motor_p.append(motor_p)
        self.delay_motor_v.append(motor_v)

        self.delay_motor_p.pop(0)
        self.delay_motor_v.pop(0)

    def sample(self, delay_low, delay_high):
        delay_motor_p = np.array(self.delay_motor_p)
        delay_motor_v = np.array(self.delay_motor_v)

        shape = delay_motor_p.shape  # (6, 12)
        i = self.np_random.randint(delay_low, delay_high, size=(shape[1],))

        sample_p = delay_motor_p[i, self.i2]

        sample_v = delay_motor_v[i, self.i2]
        return sample_p, sample_v


class robot_state_delay:
    def __init__(self, max_len, np_random):
        # ==========延迟=============
        self.robot_state_list = [None for _ in range(max_len)]
        self.g_list = [None for _ in range(max_len)]
        self.max_len = max_len
        self.np_random = np_random

    def append(self, robot_state, g):
        self.robot_state_list.append(robot_state)
        self.g_list.append(g)

        self.robot_state_list.pop(0)
        self.g_list.pop(0)

    def sample(self, delay_low, delay_high):
        i = self.np_random.randint(delay_low, delay_high)
        return self.robot_state_list[i], self.g_list[i], i
