import random
import torch
import numpy as np
from utils.functions import euler_angles_to_matrix, matrix_to_euler_angles
import gym
from Env.config import max_v, min_v
from scipy.spatial.transform import Rotation
from numba import njit

command_range = [min_v * 3 - 0.2 * 3, max_v * 3 + 0.2 * 3]
dtype = torch.double


def reset_env():
    r_ = random.uniform(0, 1)
    if r_ <= 0.1:
        use_default_terrain = True
        heightPerturbationRange = 1e-7
    elif 0.1 < r_ <= 0.2:
        use_default_terrain = False
        heightPerturbationRange = 0.025
    else:
        use_default_terrain = False
        heightPerturbationRange = random.uniform(1e-4, 0.025)
    return use_default_terrain, heightPerturbationRange


@njit(cache=True)
def fix_position(state, target_position=None, res=None):
    base_p = state[:, :2]
    if res is None:
        base_p_0 = base_p[-1, :]
        res = base_p_0 - target_position
    base_p_ = base_p - res
    state[:, :2] = base_p_
    return state, res


def fix_euler(state, frame_stack=4, target_euler=0, random_R=None):
    if random_R is None:
        R_state = state[:, 21:30].reshape(4, 3, 3)[-1]  # 提取原来的旋转矩阵
        euler = Rotation.from_matrix(R_state).as_euler('xyz')
        # now_euler = euler[-1]
        random_euler = target_euler - euler[-1]
        # euler[:, -1] = euler[:, -1] - res
        # new_R_state = euler_angles_to_matrix(euler, 'XYZ')
        # random_euler = np.array([0., 0., res], dtype=res.dtype)
        random_R = Rotation.from_euler('z', random_euler).as_matrix().reshape(1, 3, 3)
        # random_R = euler_angles_to_matrix(random_euler, 'XYZ').unsqueeze(0).expand(frame_stack, 3, 3)
    state, random_R = random_euler_state(state, None, frame_stack=frame_stack, self_z=True, random_R=random_R)
    return state, random_R


def random_euler_state(state, random_euler, frame_stack=4, self_z=False, random_R=None):
    # 注意，一定要复制
    R_state = state[:, 21:30].reshape(frame_stack, 3, 3)  # 提取原来的旋转矩阵
    if random_R is None:
        random_R = Rotation.from_euler('z', random_euler).as_matrix().reshape(1, 3, 3)
        # 处理维度
        # random_R = random_R.unsqueeze(0).expand(frame_stack, 3, 3)

    new_R_state = random_R @ R_state
    state[:, 21:30] = new_R_state.reshape(frame_stack, 9)

    if not self_z:  # 绕原点旋转
        new_pos = (random_R @ state[:, :3].reshape(frame_stack, 3, 1)).reshape(frame_stack, 3)
        state[:, :3] = new_pos
    else:  # 绕自身旋转
        current_pos = state[-1, :3].reshape(1, 3)
        new_pos = state[:, :3] - current_pos
        new_pos = (random_R @ new_pos.reshape(frame_stack, 3, 1)).reshape(frame_stack, 3) + current_pos
        state[:, :3] = new_pos
    gravity = (random_R @ state[:, 58:61].reshape(frame_stack, 3, 1)).reshape(frame_stack, 3)

    state[:, 58:61] = gravity
    return state, random_R


def random_euler_state_2(state, random_euler, frame_stack=4, self_z=False):
    # 注意，一定要复制
    # random_euler = random_euler.copy()
    # state = state.copy()
    state = torch.tensor(state.reshape(1, frame_stack, -1), dtype=dtype)
    R_state = state[:, :, 21:30].reshape(1, frame_stack, 3, 3)  # 提取原来的旋转矩阵

    # origin_base_euler = info[1]['origin_base_euler']
    # euler = 150 / 57.3 - origin_base_euler[-1]
    # random_euler[-1] = euler

    random_euler = torch.tensor(random_euler.reshape(1, -1), dtype=dtype)
    # random_euler = torch.tensor([0, 0, 45/57.3], dtype=dtype).to('cuda').reshape(1, -1)
    random_R = euler_angles_to_matrix(random_euler, 'XYZ')
    # 处理维度
    random_R = random_R.unsqueeze(1).expand(1, frame_stack, 3, 3)

    # print(1111, matrix_to_euler_angles(R_state[:, -1, :, :], 'ZYX').flip(-1))
    new_R_state = random_R @ R_state
    # print(2222, matrix_to_euler_angles(new_R_state[:, -1, :, :], 'ZYX').flip(-1))
    state[:, :, 21:30] = new_R_state.reshape(1, frame_stack, 9)

    if not self_z:  # 绕原点旋转
        new_pos = (random_R @ state[:, :, :3].unsqueeze(-1)).squeeze(-1)
        state[:, :, :3] = new_pos
    else:  # 绕自身旋转
        current_pos = state[:, -1, :3].unsqueeze(1)
        new_pos = state[:, :, :3] - current_pos
        new_pos = (random_R @ new_pos.unsqueeze(-1)).squeeze(-1) + current_pos
        state[:, :, :3] = new_pos
    gravity = (random_R @ state[:, :, 58:61].unsqueeze(-1)).squeeze(-1)

    state[:, :, 58:61] = gravity
    return state.flatten()


def angle_need_rotate(start, end):
    angle = (end - start + np.pi) % (2 * np.pi) - np.pi
    return angle


def minimal_angle_rotate(current_angle, target_angle, alpha=0.9, clip=0.1):
    angle = angle_need_rotate(start=current_angle, end=target_angle)
    current_angle = current_angle + torch.clip(angle * (1 - alpha), -clip, clip)
    current_angle = fig_angle(current_angle)
    return current_angle


def minimal_angle_rotate_np(current_angle, target_angle, alpha=0.9, clip=0.1):
    angle = angle_need_rotate(start=current_angle, end=target_angle)
    current_angle = current_angle + np.clip(angle * (1 - alpha), -clip, clip)
    current_angle = fig_angle_np(current_angle)
    return current_angle


def fig_angle(target):
    target = target - (target > 2 * np.pi).to(target.dtype) * 2 * np.pi
    target = target + (target < 0).to(target.dtype) * 2 * np.pi
    return target


def fig_angle_np(target):
    target = target - (target > 2 * np.pi).astype(target.dtype) * 2 * np.pi
    target = target + (target < 0).astype(target.dtype) * 2 * np.pi
    return target
