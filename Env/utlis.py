import time

from numba import njit
import numpy as np
from Env.config import des_p_low, des_p_high, motor_low_1, motor_high_1

x1 = 2.275
y1 = 19.56
x2 = 36
y2 = 11.0


@njit(cache=True)
def cal_motor_t_range(motor_v):
    t = (np.abs(motor_v) - y1) / (y2 - y1) * (x2 - x1) + x1
    t = np.maximum(t, 0)
    t = np.minimum(t, 36)
    return t


@njit(cache=True)
def motor(kp, des_p, current_p, kd, des_v, current_v, feedforward_t, true_motor_v):
    torque = kp * (des_p - current_p) + kd * (des_v - current_v) + feedforward_t
    max_range = cal_motor_t_range(true_motor_v)
    max_range = np.where(
        (torque * true_motor_v) > 0,
        max_range,
        36,
    )
    torque = np.maximum(torque, -max_range)
    torque = np.minimum(torque, max_range)
    return torque


@njit(cache=True)
def fig_des_p(motor_p, ):
    """
    根据关节位置,限制动作的上下限
    """
    # des_p_high_ = np.copy(des_p_high)
    # des_p_low_ = np.copy(des_p_low)
    motor_high = [1.1, 2.9, 0.2, 1.3, 2.9, 0.2, 1.1, 1.0, 2.4, 1.3, 1.0, 2.4]
    motor_low = [-1.3, -1.0, -2.4, -1.1, -1.0, -2.4, -1.3, -2.9, -0.2, -1.1, -2.9, -0.2]
    # motor_high_1 = [0.8, 2.7, -0.174, 1.0, 2.7, -0.174, 0.8, 0.5, 2.25, 1.0, 0.5, 2.25]

    p_0 = motor_p[0]
    if p_0 <= -0.:
        if abs(p_0) < 0.9:
            motor_high[3] = -(abs(p_0) / 0.9 * 1.4) + 2.0
        else:
            motor_high[3] = 0.6
    elif p_0 > 0.:
        motor_low[3] = -min(max(1.2 - abs(p_0), 0), 1.2)

    p_3 = motor_p[-3]
    if p_3 >= 0.:
        if abs(p_3) < 0.9:
            motor_high[-6] = -(abs(p_3) / 0.9 * 1.4) + 2.0
        else:
            motor_low[-6] = -0.6
    elif p_3 < -0.:
        motor_high[-6] = min(max(1.2 - abs(p_0), 0), 1.2)

    # des_p_high_ = np.clip(des_p_high_, motor_low_1, motor_high_1)
    # des_p_low_ = np.clip(des_p_low_, motor_low_1, motor_high_1)
    return motor_high, motor_low


des_p_t = np.array([8, 4, 4,
                    8, 4, 4,
                    8, 4, 4,
                    8, 4, 4, ])


@njit(cache=True)
def desp_limit(des_p, kd, delayed_v, delayed_p, kp):
    t_v = kd * - delayed_v
    t = cal_motor_t_range(delayed_v)
    # t = motor(kp, des_p, delayed_p, kd, 0, delayed_v, 0, delayed_v)
    t1 = np.sign(delayed_v) * np.minimum((t + 1.35), 33)
    t2 = -33 * np.sign(delayed_v)
    p1 = (t1 - t_v) / kp + delayed_p
    p2 = (t2 - t_v) / kp + delayed_p
    p_high = np.maximum(p1, p2)
    p_low = np.minimum(p1, p2)
    return p_high, p_low, t_v


@njit(cache=True)
def power_limit(des_p, kd, delayed_v, delayed_p, kp, torque=None, t_v=None, max_power=450):
    if t_v is None:
        t_v = kd * - delayed_v
    if torque is None:
        torque = kp * (des_p - delayed_p) + t_v
        # torque = np.maximum(torque, -36)
        # torque = np.minimum(torque, 36)
    power_ = delayed_v * torque
    power = np.maximum(power_, 0)
    sum_power = power.sum()
    if sum_power <= max_power:
        return des_p, torque, t_v
    mask = (power >= 0).astype(np.int32)
    per = max_power / sum_power
    target_torque = torque * per * mask + torque * (1 - mask)
    new_des_p = (target_torque - t_v + kp * delayed_p) / kp
    return new_des_p, target_torque, t_v


min_torque = 4
max_torque = 95


@njit(cache=True)
def torque_limit(des_p, kd, delayed_v, delayed_p, kp, torque=None, t_v=None, target=85):
    if t_v is None:
        t_v = kd * - delayed_v
    if torque is None:
        torque = kp * (des_p - delayed_p) + t_v
        # torque = np.maximum(torque, -36)
        # torque = np.minimum(torque, 36)
    positive_mask = ((torque * delayed_v) >= 0).astype(np.int32)  # 忽略期望转速和扭矩方向相反的部分
    v_mask = (np.abs(delayed_v) > 1).astype(np.int32)

    mask = positive_mask * v_mask  #
    abs_sum_up_t = np.abs(torque * mask).sum()

    if abs_sum_up_t <= target:
        return des_p

    per = target / abs_sum_up_t
    target_torque = torque * per * mask + torque * (1 - mask)
    new_des_p = (target_torque - t_v + kp * delayed_p) / kp

    return new_des_p


def cal_m_power(torque, motor_v):
    # abs_motor_v = np.abs(motor_v)
    # abs_torque = np.abs(torque)
    #
    # # 电机扭矩和转速方向相同
    # direction = torque * motor_v
    # power = np.where(direction >= 0,
    #                  abs_torque * abs_motor_v,
    #                  np.zeros_like(abs_motor_v))
    power = np.maximum(motor_v * torque, 0)
    return power


@njit(cache=True)
def matrix_x(x):
    batch = x.shape[0]
    cos_x = np.cos(x)
    sin_x = np.sin(x)

    zeros = np.zeros_like(cos_x)
    ones = np.ones_like(cos_x)

    m_x = np.concatenate((
        ones, zeros, zeros,
        zeros, cos_x, -sin_x,
        zeros, sin_x, cos_x,
    ), -1).reshape(batch, 3, 3)
    return m_x


@njit(cache=True)
def matrix_y(y):
    batch = y.shape[0]
    cos_y = np.cos(y)
    sin_y = np.sin(y)

    zeros = np.zeros_like(cos_y)
    ones = np.ones_like(sin_y)

    m_y = np.concatenate((
        cos_y, zeros, sin_y,
        zeros, ones, zeros,
        -sin_y, zeros, cos_y,
    ), -1).reshape(batch, 3, 3)
    return m_y


@njit(cache=True)
def matrix_z(z):
    batch = z.shape[0]

    cos_z = np.cos(z)
    sin_z = np.sin(z)
    zeros = np.zeros_like(cos_z)
    ones = np.ones_like(cos_z)

    m_z = np.concatenate((
        cos_z, -sin_z, zeros,
        sin_z, cos_z, zeros,
        zeros, zeros, ones,
    ), -1).reshape(batch, 3, 3)

    return m_z


@njit(cache=True)
def matrix_x_y_z(euler):
    batch = euler.shape[0]
    cos = np.cos(euler)
    sin = np.sin(euler)

    cos_x = np.expand_dims(cos[:, 0], 1)
    sin_x = np.expand_dims(sin[:, 0], 1)

    cos_y = np.expand_dims(cos[:, 1], 1)
    sin_y = np.expand_dims(sin[:, 1], 1)

    cos_z = np.expand_dims(cos[:, 2], 1)
    sin_z = np.expand_dims(sin[:, 2], 1)

    zeros = np.zeros_like(cos_z)
    ones = np.ones_like(cos_z)

    m_z = np.concatenate((
        cos_z, -sin_z, zeros,
        sin_z, cos_z, zeros,
        zeros, zeros, ones,
    ), -1).reshape(batch, 3, 3)

    m_y = np.concatenate((
        cos_y, zeros, sin_y,
        zeros, ones, zeros,
        -sin_y, zeros, cos_y,
    ), -1).reshape(batch, 3, 3)

    m_x = np.concatenate((
        ones, zeros, zeros,
        zeros, cos_x, -sin_x,
        zeros, sin_x, cos_x,
    ), -1).reshape(batch, 3, 3)

    return m_z, m_y, m_x


matrix_x_y_z(np.random.uniform(-1, 1, size=(100, 3)))
