# import time
#
import torch

import numpy as np
# from scipy.spatial.transform import Rotation
# from torch.distributions import Normal, Uniform
# from pytorch3d.transforms import matrix_to_euler_angles

euler_z_reward_weight = 0  # Z方向
xy_pos_reward_weight = 25.0  # xy位置
velocity_cost_weight = 2.5
min_Val = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_distance(pos):  # 计算xy到原点的距离
    return pos.pow(2).sum(-1, keepdim=True).sqrt()


def get_all_ob(state):
    state = state.clone()  # state = state.clone()注意这里必须要家clone(),不然会有指针问题
    base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction = \
        state[:, :, :3], \
        state[:, :, 3:15], \
        state[:, :, 15:18], \
        state[:, :, 18:21], \
        state[:, :, 21:30], \
        state[:, :, 30:42], \
        state[:, :, 42:54], \
        state[:, :, 54:58], \
        state[:, :, 58:61], \
        state[:, :, 61:62], \
        state[:, :, 62:63]
    return base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction,


# def build_none_pro_state(state, frame_stack, mean, std):
#     batch_size = state.size(0)
#     state = state.reshape(batch_size, frame_stack, -1)
#     state = (state - mean) / std
#     base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass = get_all_ob(state)
#     base_p = base_p[:, :, :2]  # 没有Z轴
#     state = torch.cat([base_p, last_action, base_w, R, motor_p, motor_v, foot_contact], -1).reshape(batch_size, -1)
#     return state


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


Random_coeffcient = {  # 各种随机系数
    'imu_orientation_bias': 0.05,  # imu欧拉角偏移(弧度)
    'imu_orientation_std': 0.05,  # imu欧拉角噪声(弧度)
    'base_w': 0.08,  # 机体角速度速度(%)
    'base_velocity': 0.08,  # 机体速度(%)
    'base_position_bias': 0.01,  # 位置误差偏移(m)
    'base_position_std': 0.01,  # 位置误差噪声(m)
    'motor_velocity': 0.08,  # 电机速度(%)
    'motor_position_bias': 0.02,  # 电机位置偏移(弧度)
    'motor_position_std': 0.02,  # 电机位置噪声(弧度)
}


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]


def axis_angle_to_matrix(axis, angle):
    size = axis.size()
    device = axis.device
    size_ = [1 for _ in size[:-1]]
    eye = torch.eye(size[-1], dtype=torch.double).to(device).reshape(size_ + [3, 3])
    eye = eye.expand(size[:-1] + (3, 3))
    cos_angle = torch.cos(angle).unsqueeze(-1)
    sin_angle = torch.sin(angle).unsqueeze(-1)

    # 这段有点蠢，但是没办法
    # 更简单的做法是u_x = torch.cross(axis, eye * -1, dim=-1)
    # 1.09不支持cross广播，没办法只能这样了
    # print(axis.size(), eye.size())
    u_x = torch.cross(axis.unsqueeze(0), eye * -1, dim=-1)
    # print(u_x)
    #
    # u_x = []
    # eye_size = len(eye.size())
    # eye_size = [-1] + [x for x in range(eye_size - 1)]
    # eye_ = (eye * -1).transpose(-1, -2).permute(*eye_size)
    # for x in range(3):
    #     u_x.append(torch.cross(axis, eye_[x], dim=-1))
    #
    # u_x = torch.cat(u_x, -1).reshape(eye.size())
    # print(u_x.size(), u_x)

    if len(size) == 1:
        u_x_2 = torch.einsum('i,j->ij', axis, axis)
    elif len(size) == 2:
        u_x_2 = torch.einsum('bi,bj->bij', axis, axis)
    elif len(size) == 3:
        u_x_2 = torch.einsum('bfi,bfj->bfij', axis, axis)
    else:
        return False
    r = cos_angle * eye + sin_angle * u_x + (1 - cos_angle) * u_x_2
    return r


def random_f_xyz(force_low, force_up, np_random):
    external_force = np_random.uniform(force_low, force_up)
    angle1 = np_random.uniform(-np.pi / 4, np.pi / 4)
    angle2 = np_random.uniform(0, np.pi * 2)
    fz = external_force * np.sin(angle1)
    fx_y = external_force * np.cos(angle1)
    fx = fx_y * np.cos(angle2)
    fy = fx_y * np.sin(angle2)
    return np.array([fx, fy, fz])
