import time

import numpy as np
from copy import deepcopy
import os
import gym
from Env.config import *
from Env.functions import euler_angles_to_matrix, axis_angle_to_matrix, matrix_to_euler_angles
import torch
from scipy.spatial.transform import Rotation
from Env.functions import random_f_xyz


class Env_random:
    def __init__(self):
        self.all_random_message = {'gravity': [0, 0, -9.815],
                                   'default_g': -9.815,
                                   'gravity_euler': np.zeros(3),  # 重力的方向
                                   # 'gravity_euler_matrix': torch.zeros(3, 3),
                                   # 'gravity_euler_2': np.zeros(3),  # 垂直于重力的平面欧拉角
                                   # 'gravity_euler_matrix_2': torch.zeros(3, 3),
                                   'external_force': [0, 0, 0],
                                   'external_pos': [0, 0, 0],
                                   # 'external_torque': [0, 0, 0],
                                   'base_mass': 5,
                                   'point_mass': np.zeros(8),
                                   'leg_mass': [0 for _ in range(12)],
                                   'global_Scaling': 1.0,
                                   'plane_Friction': 1.0,
                                   'foot_Friction': [0 for _ in range(4)],
                                   'foot_Friction_mean': 0,
                                   'base_Friction': 0,
                                   'motor_Friction': [0 for _ in motor_num],
                                   'motor_p_bias': np.zeros(motor_num),
                                   'motor_damping': np.zeros(motor_num),
                                   'motor_start_torque': np.zeros(motor_num),
                                   }
        self.Random_coefficient = deepcopy(Random_coefficient)
        self.gravity_angle = np.sin(self.Random_coefficient['gravity_angle_y'])
        self.point_mass = np.zeros(8)

    def set_Random_coefficient(self, Random_coefficient):
        self.Random_coefficient = deepcopy(Random_coefficient)
        self.gravity_angle = np.sin(self.Random_coefficient['gravity_angle_y'])

    def update_global_Scaling(self, np_random):
        # 全局缩放系数
        global_Scaling = 1 + np_random.uniform(-self.Random_coefficient['global_Scaling'],
                                               self.Random_coefficient['global_Scaling'])
        self.all_random_message['global_Scaling'] = global_Scaling

    def reset_random_message(self, np_random, base_mass_, leg_mass_):
        self.motor_damping = np_random.uniform(0.01, 0.03, size=(num_of_motor,))
        self.motor_start_torque = np_random.uniform(0.35, 0.7, size=(num_of_motor,))

        self.default_g = -9.815 * (
                1 + np_random.uniform(-self.Random_coefficient['gravity'], self.Random_coefficient['gravity']))

        self.gravity_euler = np.array([0, np.arcsin(np_random.uniform(-self.gravity_angle, self.gravity_angle)),
                                       np_random.uniform(0, 2 * np.pi)])

        # 外部作用力
        self.external_force = random_f_xyz(-self.Random_coefficient['external_force'],
                                           self.Random_coefficient['external_force'], np_random)

        # self.external_pos = np_random.uniform(-self.Random_coefficient['external_force'],
        #                                         self.Random_coefficient['external_force'],
        #                                         size=3)
        self.external_pos = np.array([np_random.uniform(-0.12, 0.12),
                                      np_random.uniform(-0.1, 0.1), np_random.uniform(-0.06, 0.06)])

        # 机体质量
        target_base_mass = base_mass_ + np_random.uniform(0, self.Random_coefficient['base_mass'])
        self.base_mass = target_base_mass * np_random.uniform(1 / 4, 3 / 4)
        point_mass_all = target_base_mass - self.base_mass
        co = np_random.uniform(0, 1, size=(8,))
        self.point_mass = co / co.sum() * point_mass_all

        # 其他质量
        self.leg_mass = np.array(leg_mass_) * \
                        (1 + np_random.uniform(-self.Random_coefficient['mass'],
                                               self.Random_coefficient['mass'], size=(num_of_motor,)))

        # 脚的摩擦系数
        self.foot_Friction = np_random.uniform(self.Random_coefficient['foot_Friction_low'],
                                               self.Random_coefficient['foot_Friction_high']) \
                             * np.array([1, 1, 1, 1])

        # 机体摩擦系数
        self.base_Friction = np_random.uniform(0.7, 0.9)

        # 电机摩擦
        self.motor_Friction = np_random.uniform(0.7, 0.9, size=(num_of_motor,))

        self.motor_p_bias = np_random.uniform(-self.Random_coefficient['motor_p_bias'],
                                              self.Random_coefficient['motor_p_bias'],
                                              size=(num_of_motor,))
        self.all_random_message['motor_start_torque'] = self.motor_start_torque
        self.all_random_message['motor_damping'] = self.motor_damping
        self.all_random_message['default_g'] = self.default_g
        self.all_random_message['gravity_euler'] = self.gravity_euler
        self.update_gravity(self.gravity_euler)
        self.all_random_message['gravity'] = self.gravity
        self.all_random_message['external_force'] = self.external_force
        self.all_random_message['external_pos'] = self.external_pos
        self.all_random_message['base_mass'] = self.base_mass
        self.all_random_message['point_mass'] = self.point_mass
        self.all_random_message['leg_mass'] = self.leg_mass
        # self.all_random_message['plane_Friction'] = self.plane_Friction
        self.all_random_message['foot_Friction'] = self.foot_Friction
        self.all_random_message['foot_Friction_mean'] = self.all_random_message['foot_Friction'].mean()
        self.all_random_message['base_Friction'] = self.base_Friction
        self.all_random_message['motor_Friction'] = self.motor_Friction
        self.all_random_message['motor_p_bias'] = self.motor_p_bias

    def update_random_message(self, np_random, base_mass_, leg_mass_, LPF=0.975, p=1 / 200):

        # 如果需要重新采样
        if np_random.binomial(1, p):
            # 重力
            self.default_g = -9.815 * (
                    1 + np_random.uniform(-self.Random_coefficient['gravity'], self.Random_coefficient['gravity']))

        if np_random.binomial(1, p):
            self.gravity_euler = np.array([0, np.arcsin(np_random.uniform(-self.gravity_angle, self.gravity_angle)),
                                           np_random.uniform(0, 2 * np.pi)])
        if np_random.binomial(1, p):
            # 外部作用力
            self.external_force = random_f_xyz(-self.Random_coefficient['external_force'],
                                               self.Random_coefficient['external_force'], np_random)

        if np_random.binomial(1, p):
            self.external_pos = np.array([np_random.uniform(-0.12, 0.12),
                                          np_random.uniform(-0.1, 0.1), np_random.uniform(-0.06, 0.06)])
        # if np_random.binomial(1, p):
        #     # 机体质量
        #     target_base_mass = base_mass_ + np_random.uniform(0, self.Random_coefficient['base_mass'])
        #     self.base_mass = target_base_mass * np_random.uniform(1 / 4, 3 / 4)
        #     point_mass_all = target_base_mass - self.base_mass
        #     co = np_random.uniform(0, 1, size=(8,))
        #     self.point_mass = co / co.sum() * point_mass_all

        # if np_random.binomial(1, p):
        #     # 其他质量
        #     self.leg_mass = np.array(leg_mass_) * \
        #                     (1 + np_random.uniform(-self.Random_coefficient['mass'],
        #                                            self.Random_coefficient['mass'], size=(num_of_motor,)))
        # if np_random.binomial(1, p):
        #     # 地面摩擦力
        #     self.plane_Friction = 1.0 + np_random.uniform(-0.05, 0.05)

        if np_random.binomial(1, p):
            # 脚的摩擦系数
            self.foot_Friction = np_random.uniform(self.Random_coefficient['foot_Friction_low'],
                                                   self.Random_coefficient['foot_Friction_high']) \
                                 * np.array([1, 1, 1, 1])

        # if np_random.binomial(1, p):
        #     # 机体摩擦系数
        #     self.base_Friction = 0.8 + np_random.uniform(-0.05, 0.05)

        # if np_random.binomial(1, p):
        #     # 电机摩擦
        #     self.motor_Friction = 0.8 + np_random.uniform(-0.05, 0.05, size=(num_of_motor,))

        for m in range(num_of_motor):
            if np_random.binomial(1, p):
                self.motor_p_bias[m] = np_random.uniform(-self.Random_coefficient['motor_p_bias'],
                                                         self.Random_coefficient['motor_p_bias'])
            if np_random.binomial(1, p):
                self.motor_damping[m] = np_random.uniform(0.01, 0.03)
            if np_random.binomial(1, p):
                self.motor_start_torque[m] = np_random.uniform(0.35, 0.7)

        # 一定要重新计算一次重力
        self.all_random_message['default_g'] = LPF * self.all_random_message[
            'default_g'] + (1 - LPF) * self.default_g
        self.all_random_message['gravity_euler'][1] = LPF * self.all_random_message[
            'gravity_euler'][1] + (1 - LPF) * self.gravity_euler[1]

        self.all_random_message['gravity_euler'][2] = minimal_angle_rotate(
            self.all_random_message['gravity_euler'][2],
            self.gravity_euler[2],
            alpha=LPF, clip=10)
        # print(self.all_random_message['gravity_euler'], self.gravity_euler)

        self.update_gravity(self.all_random_message['gravity_euler'])

        self.all_random_message['gravity'] = self.gravity

        self.all_random_message['external_force'] = LPF * self.all_random_message[
            'external_force'] + (1 - LPF) * self.external_force
        self.all_random_message['external_pos'] = LPF * self.all_random_message[
            'external_pos'] + (1 - LPF) * self.external_pos
        # self.all_random_message['base_mass'] = LPF * self.all_random_message['base_mass'] + (
        #         1 - LPF) * self.base_mass
        # self.all_random_message['point_mass'] = LPF * self.all_random_message['point_mass'] + (
        #         1 - LPF) * self.point_mass
        # self.all_random_message['leg_mass'] = LPF * self.all_random_message['leg_mass'] + (1 - LPF) * self.leg_mass
        # self.all_random_message['plane_Friction'] = LPF * self.all_random_message[
        #     'plane_Friction'] + (1 - LPF) * self.plane_Friction
        self.all_random_message['foot_Friction'] = LPF * self.all_random_message[
            'foot_Friction'] + (1 - LPF) * self.foot_Friction
        self.all_random_message['foot_Friction_mean'] = self.all_random_message['foot_Friction'].mean()
        # self.all_random_message['base_Friction'] = LPF * self.all_random_message[
        #     'base_Friction'] + (1 - LPF) * self.base_Friction
        # self.all_random_message['motor_Friction'] = LPF * self.all_random_message[
        #     'motor_Friction'] + (1 - LPF) * self.motor_Friction
        self.all_random_message['motor_p_bias'] = LPF * self.all_random_message[
            'motor_p_bias'] + (1 - LPF) * self.motor_p_bias
        self.all_random_message['motor_damping'] = LPF * self.all_random_message[
            'motor_damping'] + (1 - LPF) * self.motor_damping
        self.all_random_message['motor_start_torque'] = LPF * self.all_random_message[
            'motor_start_torque'] + (1 - LPF) * self.motor_start_torque

    def apply(self, pybullet_client, quadruped, terrain):
        # global_Scaling, external_force, motor_torque_bias不在这里用

        # 重力
        pybullet_client.setGravity(*self.all_random_message['gravity'].tolist())  # 设置重力

        # 机体质量, 机体摩擦
        pybullet_client.changeDynamics(quadruped, BASE_LINK_ID, mass=self.all_random_message['base_mass'],
                                       lateralFriction=self.all_random_message['base_Friction'])
        for x in range(num_of_point):
            pybullet_client.changeDynamics(quadruped, point_num[x], mass=self.all_random_message['point_mass'][x])

        # 其他质量, 摩擦
        for x in range(num_of_motor):
            pybullet_client.changeDynamics(quadruped, motor_num[x],
                                           mass=self.all_random_message['leg_mass'][x],
                                           lateralFriction=self.all_random_message['motor_Friction'][x],
                                           jointDamping=self.all_random_message['motor_damping'][x])

        # # 机体摩擦
        # pybullet_client.changeDynamics(quadruped, BASE_LINK_ID,
        #                                lateralFriction=self.all_random_message['base_Friction'])
        # for x in range(num_of_motor):
        #     pybullet_client.changeDynamics(quadruped, motor_num[x],
        #                                    lateralFriction=self.all_random_message['motor_Friction'][x])

        # 脚的摩擦系数
        for x in range(len(foot_num)):
            pybullet_client.changeDynamics(quadruped, foot_num[x],
                                           lateralFriction=self.all_random_message['foot_Friction'][x])
        # 设置地面的摩擦系数
        pybullet_client.changeDynamics(terrain, -1, lateralFriction=self.all_random_message['plane_Friction'])

    def get_all_random(self):
        return self.all_random_message

    def update_gravity(self, gravity_euler):
        # axis = torch.DoubleTensor([np.cos(gravity_euler[-1]), np.sin(gravity_euler[-1]), 0])
        # m2 = axis_angle_to_matrix(axis, torch.tensor(gravity_euler[1], dtype=torch.double))
        # self.all_random_message['m2'] = m2.clone()

        n_m2 = Rotation.from_rotvec(
            gravity_euler[1] * np.array([np.cos(gravity_euler[-1]), np.sin(gravity_euler[-1]), 0])).as_matrix()
        # print(n_m2 - m2.numpy())
        self.all_random_message['m2'] = n_m2

        # g = (n_m2 @ np.array([0, 0, self.all_random_message['default_g']]).reshape(3, 1)).reshape(3)
        # g = g.tolist()
        self.gravity = (n_m2 @ np.array([0, 0, self.all_random_message['default_g']]).reshape(3, 1)).flatten()

        # l = (m2 @ torch.DoubleTensor([0, 0, 1]).reshape(3, 1)).flatten()     # 新平面法向量
        #
        # angle = torch.DoubleTensor([gravity_euler[1]])  # 求出旋转轴
        # # 注意叉乘顺序
        # axis_ = torch.cross(l, torch.DoubleTensor([0, 0, 1]), -1)  # 叉乘
        # axis_ = axis_ / axis_.norm()  # 转成单位向量
        # m2 = axis_angle_to_matrix(axis_, angle)
        # self.all_random_message['gravity_euler_2'] = matrix_to_euler_angles(m2, 'ZYX').flip(-1)
        # # print(111, gravity_euler, g, self.all_random_message['gravity_euler_2'])
        # self.all_random_message['gravity_euler_matrix_2'] = m2


def angle_need_rotate(start, end):
    angle = (end - start + np.pi) % (2 * np.pi) - np.pi
    return angle


def minimal_angle_rotate(current_angle, target_angle, alpha=0.9, clip=0.05):
    angle = angle_need_rotate(start=current_angle, end=target_angle)
    current_angle = current_angle + np.clip(angle * (1 - alpha), -clip, clip)
    current_angle = fig_angle(current_angle)
    return current_angle


def fig_angle(target):
    target = target - (target > 2 * np.pi).astype(target.dtype) * 2 * np.pi
    target = target + (target < 0).astype(target.dtype) * 2 * np.pi
    return target
