import time
#
import torch
from Env.config import max_v, min_v
import numpy as np

dtype = torch.double
# from scipy.spatial.transform import Rotation
# from torch.distributions import Normal, Uniform
# from pytorch3d.transforms import matrix_to_euler_angles

# euler_z_reward_weight = 0  # Z方向
# xy_pos_reward_weight = 120.0  # xy位置
velocity_cost_weight = 2.5
min_Val = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# max_v = torch.FloatTensor([2.]).to('cuda')

def huber_reward_fun(value, c1, c2):
    v = torch.abs(value / c1)
    r = torch.where(v <= 1, - v ** 2 / 2, - (v - 0.5))
    return r * c2


def huber_reward_fun2(value, c1, c2):
    v = torch.abs(value / c1)
    delta = np.sqrt(1 / c2)
    r = torch.where(v <= delta, - v ** 2 / 2, - delta * (v - 0.5 * delta))
    return r * c2


# def exp_loss(value, c1):
#     loss = torch.exp(-value ** 2 * c1)
#     return loss

#@torch.compile
def exp_loss(value, c1, l=0.9):
    sign = torch.sign(value)
    a = l * sign
    loss = torch.where(
        torch.abs(value) <= l,
        torch.exp(-value ** 2 * c1),
        (-2 * c1 * a * (value - a) + 1) * torch.exp(-a ** 2 * c1)
    )
    # loss = np.exp(-value ** 2 * c1)
    return loss


per = 0.05
per2 = (1 / per - 0.5)


#@torch.compile
def update_pos_z_direct_reward(reward, base_v_w, command_p, base_v, this_time):
    """
    last_base_p, next_base_p必须是缩放前的真实位置,且不包含z轴
    """
    # last_base_p = last_base_p[:, :2]
    # next_base_p = next_base_p[:, :2]
    # last_distance = cal_L2_norm(last_base_p)
    # current_distance = cal_L2_norm(next_base_p)
    #
    # v = (last_distance - current_distance) / this_time
    # vt = ((next_base_p - last_base_p) ** 2).sum(-1, keepdim=True)
    # v_h = torch.sqrt(torch.clip(vt / (this_time ** 2) - v ** 2, 0, np.inf))
    _, _, target_d, cos_v_p, sin_v_p, v_total = cal_velocity(command_p, base_v_w[:, :2])
    target_v = torch.clip(target_d / 3, min_v, max_v)

    # ==================================================

    forward_v = base_v[:, 0].reshape(-1, 1)

    # ==================================================
    # other_v = base_v[:, -1].reshape(-1, 1)
    cos_base_v = base_v[:, 1].reshape(-1, 1)
    # p_reward = (1.0 - current_distance) * 0.3 * (current_distance <= 1.0).float()
    # v_cost = huber_reward_fun2(v_h, total_v / 3, 0.4)
    # flag = (cos_v_p <= 0).to(dtype)

    sin_v_p = torch.where(
        cos_v_p <= 0,
        1,
        sin_v_p,
    )
    v_cost = -sin_v_p ** 2 * 1.25

    # p = target_v * per
    xy_pos_co = exp_loss(target_v - v_total, 3.0)
    # xy_pos_co = huber_reward_fun(target_v - v_total, target_v * per, 1) / per2 + 1
    xy_pos_reward = 1.50 * xy_pos_co * cos_v_p

    # ==================================================

    # y_velocity_co = huber_reward_fun(target_v - forward_v, target_v * per, 1) / per2 + 1
    y_velocity_co = exp_loss(target_v - forward_v, 3.0)
    y_velocity_cost = 1.50 * y_velocity_co * cos_base_v
    # ======================
    # r = torch.where(target_d / 3 <= max_v,
    #                 1.50 * (huber_reward_fun(target_v - v_p, target_v * per, 1) / per2 + 1) +
    #                 1.50 * (huber_reward_fun(target_v - forward_v, target_v * per, 1) / per2 + 1),
    #                 (v_p + forward_v) * 1.50 / max_v
    #                 )

    return reward + xy_pos_reward + y_velocity_cost + v_cost


#@torch.compile
def cal_velocity(base_p, v_w):
    # 计算世界坐标系的速度
    distance = cal_L2_norm(base_p)

    # 注意这里cos是剪掉pi的
    cos_sin_base_p = - base_p / distance

    # 速度方向的cos
    v_total = cal_L2_norm(v_w)
    cos_sin_v_w = torch.where(v_total <= min_Val,
                              v_w * 0,
                              v_w / v_total
                              )
    # cos_sin_v_w = v_w / (v_total + min_Val)

    # 计算在位置方向上的速度分量
    cos_v_p = (cos_sin_v_w * cos_sin_base_p).sum(-1, keepdim=True)
    sin_v_p = cos_sin_v_w[:, 1] * cos_sin_base_p[:, 0] - cos_sin_v_w[:, 0] * cos_sin_base_p[:, 1]
    sin_v_p = sin_v_p.unsqueeze(-1)
    # v_p = v_total * cos_v_p
    # v_other = v_total * sin_v_p
    return 0, 0, distance, cos_v_p, sin_v_p, v_total


def update_velocity_cost(batch_size, reward, next_base_p, next_base_v, R):
    """
    last_base_p, next_base_p必须是缩放前的真实位置,且不包含z轴
    """
    v_w = (R @ next_base_v.reshape(batch_size, 3, 1)).reshape(batch_size, 3)[:, :2]  # 只管xy不管z
    distance = cal_L2_norm(next_base_p)

    # 注意这里cos是剪掉pi的
    cos_sin_base_p = - next_base_p / (distance + min_Val)

    # 速度方向的cos
    v_total = v_w.pow(2).sum(-1, keepdim=True).sqrt()
    cos_sin_v_w = v_w / (v_total + min_Val)

    # 计算在位置方向上的速度分量
    cos_v_p = (cos_sin_v_w * cos_sin_base_p).sum(-1, keepdim=True)
    # sin_v_p = cos_sin_v_w[:, 1] * cos_sin_base_p[:, 0] - cos_sin_v_w[:, 0] * cos_sin_base_p[:, 1]
    v_p = v_total * cos_v_p
    # v_other = v_total * sin_v_p.unsqueeze(-1)

    # c = torch.clip(distance, 0, 2) / 2

    reward += velocity_cost_weight * v_p
    return reward


def cal_L2_norm(pos):  # 计算xy到原点的距离
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


def reshape_samples(samples, batch_size):
    state, action, next_state, reward, not_done = samples
    return (
        state.reshape(batch_size, -1),
        action,
        next_state.reshape(batch_size, -1),
        reward,
        not_done,
    )


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


# =====前后互换=================

index = [9, 10, 11,
         6, 7, 8,
         3, 4, 5,
         0, 1, 2,
         ]
double_index = index + [x + 12 for x in index]

inverse_a = torch.tensor([-1, -1, 1])


def inverse_state(state, frame_stack=4, horizon=False, euler=None, cat=True):  # 根据当前坐标更新所有state
    if type(state) == tuple:
        base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction = state
        device = base_p.device
        batch_size = base_p.size(0)
    else:
        device = state.device
        batch_size = state.size(0)

        state = state.reshape(batch_size, frame_stack, -1)
        base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction = get_all_ob(
            state)

    a = inverse_a.to(device)
    if euler is None:
        euler = matrix_to_euler_angles(R.reshape(batch_size, frame_stack, 3, 3), 'ZYX').flip(-1)
    euler = euler * a[None, None, :]  # 欧拉角取反

    # base_p = (reflection_m @ base_p.unsqueeze(-1)).squeeze(-1)
    base_p = base_p * a[None, None, :]
    last_action = last_action[:, :, index] * -1
    base_v = base_v * a[None, None, :]
    base_w = base_w * a[None, None, :]
    # print(euler_angles_to_matrix(euler, 'XYZ').size())
    R = euler_angles_to_matrix(euler, 'XYZ')
    R = R.reshape(batch_size, frame_stack, 9)
    # print(222, R)
    # time.sleep(10000)
    motor_p = motor_p[:, :, index] * -1
    motor_v = motor_v[:, :, index] * -1
    foot_contact = foot_contact[:, :, [3, 2, 1, 0]]
    # ============

    # 重力
    gravity = gravity * a[None, None, :]
    if cat:
        state = torch.cat([base_p, last_action, base_v, base_w, R, motor_p, motor_v,
                           foot_contact, gravity, mass, foot_Friction], -1)
        state = state.reshape(batch_size, -1)
        return state, euler
    else:
        return (base_p, last_action, base_v, base_w, R, motor_p, motor_v,
                foot_contact, gravity, mass, foot_Friction), euler


def inverse_action(action, horizon=False):  # 翻转
    return action[index] * - 1


# =====左右互换,解决左右撇子问题=================


index2 = [3, 4, 5,
          0, 1, 2,
          9, 10, 11,
          6, 7, 8, ]
aaa = [-1, 1, 1,
       -1, 1, 1,
       -1, 1, 1,
       -1, 1, 1, ]


def reflect_state(state, frame_stack=4, horizon=False, euler=None, cat=True):  # 左右互换
    if type(state) == tuple:
        base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction = state
        device = base_p.device
        batch_size = base_p.size(0)
    else:
        device = state.device
        batch_size = state.size(0)

        state = state.reshape(batch_size, frame_stack, -1)
        base_p, last_action, base_v, base_w, R, motor_p, motor_v, foot_contact, gravity, mass, foot_Friction = get_all_ob(
            state)

    reflect_a = torch.tensor([1, -1, 1]).to(device)
    reflect_b = torch.tensor([-1, 1, -1]).to(device)
    reflect_aaa = torch.tensor(aaa)[None, None, :].to(device)

    if euler is None:
        euler = matrix_to_euler_angles(R.reshape(batch_size, frame_stack, 3, 3), 'ZYX').flip(-1)
    euler = euler * reflect_b[None, None, :]  # 欧拉角取反

    base_p = base_p * reflect_a[None, None, :]
    last_action = last_action[:, :, index2] * reflect_aaa
    base_v = base_v * reflect_a[None, None, :]
    base_w = base_w * reflect_b[None, None, :]

    R = euler_angles_to_matrix(euler, 'XYZ')
    R = R.reshape(batch_size, frame_stack, 9)
    # print(222, R)
    # time.sleep(10000)
    motor_p = motor_p[:, :, index2] * reflect_aaa
    motor_v = motor_v[:, :, index2] * reflect_aaa
    foot_contact = foot_contact[:, :, [1, 0, 3, 2]]
    # ============

    # 重力
    gravity = gravity * reflect_a[None, None, :]
    if cat:
        state = torch.cat([base_p, last_action, base_v, base_w, R, motor_p, motor_v,
                           foot_contact, gravity, mass, foot_Friction], -1)
        state = state.reshape(batch_size, -1)
        return state, euler
    else:
        return (base_p, last_action, base_v, base_w, R, motor_p, motor_v,
                foot_contact, gravity, mass, foot_Friction), euler


def reflect_action(action, horizon=False):  # 翻转
    return action[index2] * aaa
