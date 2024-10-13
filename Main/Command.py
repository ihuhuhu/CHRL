# import random
# import torch
# import numpy as np
# from utils.functions import euler_angles_to_matrix, matrix_to_euler_angles
# import gym
import time

import torch

from Main.Command_function import *

dtype = torch.double


class Base_Command:
    def __init__(self, frame_stack):
        self.count = 0
        self.frame_stack = frame_stack

    def change_state(self, state, info, self_z):
        self.count += 1
        state = self.change_state_(state.copy(), info, self_z)
        return state

    def change_state_(self, state, info, self_z):
        return state

    def build_env(self, env_name, urdf_root, GUI, use_ex_f, use_default_terrain, heightPerturbationRange):
        env = gym.make(env_name, urdf_root=urdf_root, GUI=GUI,
                       seed=random.randint(0, 100000),
                       teacher_frame_stack=self.frame_stack,
                       use_ex_f=use_ex_f,
                       use_default_terrain=use_default_terrain,
                       heightPerturbationRange=heightPerturbationRange)  # 构造环境
        return env

    def store(self, state, next_state):
        return state, next_state

    def all_count(self):
        return self.count


class Uniform_Command(Base_Command):
    def __init__(self, frame_stack):
        super(Uniform_Command, self).__init__(frame_stack)
        self.res = np.zeros(2)

        self.store_state = None
        self.random_R = np.eye(3)[None, :, :]
        self.init()

    def init(self):
        self.reset_p()
        self.random_distance = self.target_distance
        self.random_pos = np.array([self.random_distance, 0])

        self.reset_euler()
        self.random_euler = self.target_random_euler

    def reset_p(self):
        self.target_distance = random.uniform(*command_range)

    # def reset_p_angle(self):
    #     self.target_angle = np.array([random.uniform(0, 2 * np.pi)])

    def reset_euler(self, info=None):
        self.target_random_euler = random.uniform(-np.pi, np.pi)

    def LPF(self, alpha=0.95):
        # angle = angle_need_rotate(start=self.random_euler[-1], end=self.target_random_euler[-1])
        # self.random_euler[-1] = self.random_euler[-1] + torch.clip(angle * (1 - alpha), -0.05, 0.05)
        # self.random_euler = fig_angle(self.random_euler)
        self.random_euler = minimal_angle_rotate_np(self.random_euler, self.target_random_euler, alpha)

        # angle = angle_need_rotate(start=self.random_angle, end=self.target_angle)
        # self.random_angle = self.random_angle + torch.clip(angle * (1 - alpha), -0.05, 0.05)
        # self.random_angle = fig_angle(self.random_angle)
        # self.random_angle = minimal_angle_rotate(self.random_angle, self.target_angle, alpha)

        self.random_distance = self.random_distance * alpha + self.target_distance * (1 - alpha)

        # self.random_pos = torch.tensor([self.random_distance * torch.cos(self.random_angle),
        #                                 self.random_distance * torch.sin(self.random_angle)], dtype=dtype)
        self.random_pos = np.array([self.random_distance, 0])

    def change_state_(self, state, info, self_z):
        state = state.reshape(self.frame_stack, -1)
        if np.random.binomial(1, 1/150):
            self.reset_p()
        # if random.uniform(0, 1) <= 1 / 150:
        #     self.reset_p_angle()
        if np.random.binomial(1, 1/150):
            self.reset_euler()

        self.LPF()

        state, self.res = fix_position(state, target_position=self.random_pos)
        # self.store_state = state.clone().flatten().numpy()
        # state, random_R = random_euler_state(state,
        #                                      torch.cat([torch.zeros(2, dtype=dtype), self.random_angle], -1),
        #                                      self.frame_stack, False)
        state, self.random_R = random_euler_state(state, self.random_euler, self.frame_stack, self_z=True)
        # self.random_R = random_R
        self.store_state = state.flatten()
        return state

    def build_env(self, env_name, urdf_root, GUI, use_ex_f, use_default_terrain, heightPerturbationRange):
        self.init()
        env = gym.make(env_name, urdf_root=urdf_root, GUI=GUI,
                       seed=random.randint(0, 100000),
                       teacher_frame_stack=self.frame_stack,
                       init_position_high=0.1,
                       init_position_low=0,
                       # reach_time=2000,
                       max_distance=25,
                       max_episode_steps=900,
                       use_ex_f=use_ex_f,
                       use_default_terrain=use_default_terrain,
                       heightPerturbationRange=heightPerturbationRange)  # 构造环境
        return env

    def store(self, state, next_state):
        next_state = next_state.copy().reshape(self.frame_stack, -1)
        next_state, _ = fix_position(next_state, target_position=self.random_pos)
        next_state, _ = random_euler_state(next_state, None, self.frame_stack, self_z=True,
                                           random_R=self.random_R)
        # next_state = next_state.flatten().numpy()
        # print(self.store_state.reshape(4, -1)[-1, :2], last_base_p, base_p)
        # print(self.store_state.reshape(4, -1)[:, :3], next_state.reshape(4, -1)[:, :3])
        return self.store_state, next_state.flatten()


class Turn_Command(Base_Command):
    def __init__(self, frame_stack):
        super(Turn_Command, self).__init__(frame_stack)

    def change_state_(self, state, random_euler, use_self_z):
        state = random_euler_state_2(state, random_euler, frame_stack=self.frame_stack,
                                     self_z=use_self_z)
        return state


class Uniform_Turn_Command(Uniform_Command):
    def __init__(self, frame_stack):
        super(Uniform_Turn_Command, self).__init__(frame_stack)

    def init(self):
        self.reset_p()

        self.random_distance = self.target_distance
        self.random_pos = np.array([self.random_distance, 0])

        self.reset_euler()
        self.random_euler = self.target_random_euler

    # def cal_target_v(self):
    #     self.target_v = np.clip(self.random_distance / 3, 0.8, 3.5)

    def euler_range(self):
        self.range = (self.random_distance - command_range[0]) / (command_range[1] - command_range[0]) * (
                np.pi / 1.5 - 1) + 1

    def reset_euler(self, info=None):
        self.target_random_euler = random.uniform(-1.3, 1.3)

    def LPF(self, alpha=0.95):
        # self.random_angle = minimal_angle_rotate(self.random_angle, self.target_angle, alpha)
        self.random_distance = self.random_distance * alpha + self.target_distance * (1 - alpha)

        # self.random_pos = torch.tensor([self.random_distance * torch.cos(self.random_angle),
        #                                 self.random_distance * torch.sin(self.random_angle)], dtype=dtype)
        self.random_pos = np.array([self.random_distance, 0])
        # self.cal_target_v()
        # self.euler_range()

        self.random_euler = self.random_euler * alpha + self.target_random_euler * (1 - alpha)

    def change_state_(self, state, info, self_z):
        state = state.reshape(self.frame_stack, -1)
        if np.random.binomial(1, 1/200):
            self.reset_p()

        # if random.uniform(0, 1) <= 1 / 150:
        #     self.reset_p_angle()

        if np.random.binomial(1, 1/150):
            self.reset_euler()

        self.LPF()

        state, self.res = fix_position(state, target_position=self.random_pos)
        # self.store_state = state.clone().flatten().numpy()
        # state, random_R = random_euler_state(state,
        #                                      torch.cat([torch.zeros(2, dtype=dtype), self.random_angle], -1),
        #                                      self.frame_stack, False)
        state, self.random_R = fix_euler(state, target_euler=self.random_euler)
        self.store_state = state.flatten()
        return state

    def store(self, state, next_state):
        next_state = next_state.copy().reshape(self.frame_stack, -1)
        next_state, _ = fix_position(next_state, target_position=self.random_pos)
        next_state, _ = fix_euler(next_state, target_euler=None, random_R=self.random_R)
        # next_state = next_state.flatten().numpy()
        # print(self.store_state.reshape(4, -1)[-1, :2], last_base_p, base_p)
        return self.store_state, next_state.flatten()


class All_Command:
    def __init__(self, frame_stack):
        self.commands = [
            # Base_Command(frame_stack),
            # Turn_Command(frame_stack),
            Uniform_Command(frame_stack),
            Uniform_Turn_Command(frame_stack)
        ]

        self.current_command_id = -1

    def choose_command(self):
        if self.current_command_id == -1:
            self.current_command_id = random.randint(0, len(self.commands) - 1)
            return
        count = self.get_current_count()
        count[1] = count[1] * 2
        self.current_command_id = count.index(min(count))

    def change_state(self, state, info, self_z):
        state = self.commands[self.current_command_id].change_state(state, info, self_z)
        return state

    def store(self, state, next_state):
        return self.commands[self.current_command_id].store(state, next_state)

    def build_env(self, env_name, urdf_root, GUI, use_ex_f, use_default_terrain, heightPerturbationRange):
        self.choose_command()
        # print(1233455, self.current_command_id)
        env = self.commands[self.current_command_id].build_env(env_name, urdf_root, GUI, use_ex_f,
                                                               use_default_terrain, heightPerturbationRange)
        return env

    def get_current_count(self):
        count = []
        for c in self.commands:
            count.append(c.all_count())
        return count

    def get_message(self):
        return self.commands[self.current_command_id].random_pos, \
               self.commands[self.current_command_id].random_R[-1, :, :]
