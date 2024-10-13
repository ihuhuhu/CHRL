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


# @njit(cache=True)  # numba 加速
# def fig_feedforward_t(kp, kd, des_p, motor_p, des_v, motor_v, feedforward_t):  # 抑制电机在非法工况下的位置惨差
#     # 工况两点(3, 20), (30, 2)
#     x1 = 4
#     y1 = 20
#
#     x2 = 30
#     y2 = 2.67
#
#     torque = kp * (des_p - motor_p) + kd * (des_v - motor_v)  # 先不管前馈
#
#     abs_motor_v = np.abs(motor_v)
#
#     # ================================
#     motor_v_mask = (motor_v > 0).astype(np.int32)  # 判定电机转速正反
#
#     mask_1 = (abs_motor_v > y1).astype(np.int32)  # 转速超过20
#
#     feedforward_t_1_1 = np.minimum(feedforward_t, x1 - torque)
#     feedforward_t_1_2 = np.maximum(feedforward_t, -x1 - torque)
#     feedforward_t_1 = feedforward_t_1_1 * motor_v_mask + feedforward_t_1_2 * (1 - motor_v_mask)
#
#     # ================================
#     t = (abs_motor_v - y1) / (y2 - y1) * (x2 - x1) + x1  # 该转速下的极限扭矩
#
#     feedforward_t_2_1 = np.minimum(feedforward_t, t - torque)
#     feedforward_t_2_2 = np.maximum(feedforward_t, -t - torque)
#     feedforward_t_2 = feedforward_t_2_1 * motor_v_mask + feedforward_t_2_2 * (1 - motor_v_mask)
#
#     # ================================
#     mask_3 = (abs_motor_v < y2).astype(np.int32)  # 转速超过20
#
#     feedforward_t_3_1 = np.minimum(feedforward_t, x2 - torque)
#     feedforward_t_3_2 = np.maximum(feedforward_t, -x2 - torque)
#     feedforward_t_3 = feedforward_t_3_1 * motor_v_mask + feedforward_t_3_2 * (1 - motor_v_mask)
#
#     mask_2 = ((motor_v * (torque + feedforward_t)) > 0).astype(np.int32)  # 扭矩+前馈和转速是否同向
#
#     feedforward_t = feedforward_t * (1 - mask_2) + \
#                     mask_2 * feedforward_t_1 * mask_1 + \
#                     mask_2 * feedforward_t_2 * (1 - mask_1) * (1 - mask_3) + \
#                     mask_2 * feedforward_t_3 * (1 - mask_1) * mask_3
#     # print(222, feedforward_t)
#     des_torque = torque + feedforward_t  # 预期的扭矩
#
#     # 截断扭矩,约束在+-33NM内
#     des_torque_ = np.maximum(des_torque, -33)
#     des_torque_ = np.minimum(des_torque_, 33)
#     feedforward_t -= (des_torque - des_torque_)  # 修正feedforward_t
#     des_torque = torque + feedforward_t
#     return feedforward_t, des_torque


@njit(cache=True)  # numba 加速
def fig_feedforward_t(kp, kd, des_p, motor_p, des_v, motor_v, feedforward_t):  # 抑制电机在非法工况下的位置惨差
    # 工况两点(3, 20), (30, 2)
    x1 = 4
    y1 = 20

    x2 = 30
    y2 = 2.67

    torque = kp * (des_p - motor_p) + kd * (des_v - motor_v)  # 先不管前馈

    abs_motor_v = np.abs(motor_v)

    # ================================
    t = (abs_motor_v - y1) / (y2 - y1) * (x2 - x1) + x1  # 该转速下的极限扭矩

    # ================================
    feedforward_t = np.where((motor_v * (torque + feedforward_t)) > 0,
                             np.where(abs_motor_v > y1,
                                      np.where(motor_v > 0, np.minimum(feedforward_t, x1 - torque),
                                               np.maximum(feedforward_t, -x1 - torque)),
                                      np.where(abs_motor_v < y2,
                                               np.where(motor_v > 0,
                                                        np.minimum(feedforward_t, x2 - torque),
                                                        np.maximum(feedforward_t, -x2 - torque)),
                                               np.where(motor_v > 0,
                                                        np.minimum(feedforward_t, t - torque),
                                                        np.maximum(feedforward_t, -t - torque)
                                                        )
                                               )
                                      ),
                             feedforward_t,
                             )
    des_torque = torque + feedforward_t  # 预期的扭矩

    # 截断扭矩,约束在+-33NM内
    des_torque_ = np.maximum(des_torque, -33)
    des_torque_ = np.minimum(des_torque_, 33)
    feedforward_t -= (des_torque - des_torque_)  # 修正feedforward_t
    des_torque = torque + feedforward_t
    return feedforward_t, des_torque


@njit(cache=True)
def fig_power(des_torque, motor_v, feedforward_t):  # 对功耗进行限制
    power, _, efficient = cal_power(des_torque, motor_v)
    abs_motor_v = np.abs(motor_v)
    motor_v_mask = (motor_v > 0).astype(np.int32)  # 判定电机转速正反

    mask_2 = ((motor_v * des_torque) > 0).astype(np.int32)  # 扭矩+前馈和转速是否同向

    # 总功率不得>1200
    # 平均每个电机应该减少的扭矩
    power_p = 1 - 1300 / (power.sum() + 1e-17)
    power_p = np.maximum(power_p, 0)
    target_power = power * power_p  # 下需要减少的目标功率
    t = target_power * efficient / np.maximum(abs_motor_v, 0.1)

    # power_p = np.maximum(power_p, 0)
    # target_power = power * power_p  # 下需要减少的目标功率
    # t = target_power * efficient / np.maximum(abs_motor_v, 0.1)

    # 否则就要减少前馈
    # 注意处理扭矩方向
    feedforward_t = feedforward_t * (1 - mask_2) + \
                    ((feedforward_t - t) * motor_v_mask + (feedforward_t + t) * (1 - motor_v_mask)) * mask_2

    return feedforward_t


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
def fig_des_p_2(delayed_p, delayed_v, kp, kd, feedforward_t, max_t_, t_v=None):
    # print(max_t_)
    # 首先计算当前速度对应的可用扭矩(22, 0), (0, 33.5)
    abs_v = np.abs(delayed_v)
    max_t = np.maximum((-abs_v / 22 + 1) * 34, 0)  # 扭矩不能低于0
    max_t = np.minimum(max_t, max_t_)  # 可用扭矩限制为25NM，前进agent没有这个限制

    # 然后处理速度方向，减速方向的最小扭矩是3NM
    up_t = np.where(delayed_v > 0, max_t, np.maximum(max_t, des_p_t))
    low_t = np.where(delayed_v > 0, np.minimum(-max_t, -des_p_t), -max_t)

    # 根据kp * (des_p - current_p) + kd * (des_v - current_v)计算可用des_p的区间
    if t_v is None:
        t_v = kd * - delayed_v + feedforward_t
    up_des_p = (up_t - t_v) / kp + delayed_p  # des_p的上限
    low_des_p = (low_t - t_v) / kp + delayed_p  # des_p的下限
    # up_des_p += 0.075
    up_des_p = np.minimum(33 / kp + delayed_p, up_des_p)
    # low_des_p -= 0.075
    low_des_p = np.maximum(-33 / kp + delayed_p, low_des_p)
    # print((up_des_p - delayed_p)[1], (low_des_p - delayed_p)[1], delayed_v[1])

    return up_des_p, low_des_p, t_v


@njit(cache=True)
def single_power_limit(delayed_p, delayed_v, kp, kd, feedforward_t, max_t=33.5):
    abs_v = np.abs(delayed_v) + 1e-17
    sign = np.sign(delayed_v)
    t1 = np.maximum(np.minimum(100 / abs_v, max_t), -max_t) * sign
    t2 = np.maximum(np.minimum(-100 / abs_v, max_t), -max_t) * sign
    up_t = np.maximum(t1, t2)
    low_t = np.minimum(t1, t2)
    t_v = kd * - delayed_v + feedforward_t

    up_des_p = (up_t - t_v) / kp + delayed_p  # des_p的上限
    low_des_p = (low_t - t_v) / kp + delayed_p  # des_p的下限
    return up_des_p, low_des_p, t_v


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


# @njit(cache=True)
# def torque_limit(des_p, t_v, delayed_p, kp):
#     up_t = kp * (des_p - delayed_p) + t_v
#     abs_t = np.abs(up_t)
#     abs_sum_up_t = abs_t.sum()
#     if abs_sum_up_t <= max_torque:
#         return des_p
#     t_to_cut = abs_sum_up_t - max_torque  # 需要削减的总扭矩
#
#     t_min_torque = np.maximum(up_t, -min_torque)
#     t_min_torque = np.minimum(t_min_torque, min_torque)
#     t1 = up_t - t_min_torque  # 提取>5的扭矩部分
#
#     # >5的扭矩部分，削减这部分来让总扭矩小于目标扭矩
#     per = 1 - t_to_cut / np.abs(t1).sum()
#
#     t2 = t1 * per
#     target_torque = t_min_torque + t2
#     new_des_p = (target_torque - t_v + kp * delayed_p) / kp
#
#     # new_t = kp * (new_des_p - delayed_p) + t_v
#     # print((des_p - delayed_p)[1], (new_des_p - delayed_p)[1], t_v[1], per, up_t[1], new_t[1])
#     # print(np.abs(new_t).sum())
#     # time.sleep(0.1)
#
#     # print(per, np.abs(new_t).sum())
#     # time.sleep(0.01)
#     # if np.abs(new_t).sum() > (max_torque + 0.01):
#     #     print(111111111111)
#     return new_des_p


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
def cal_power(torque, motor_v):
    abs_motor_v = np.abs(motor_v)
    abs_torque = np.abs(torque)

    # 电机扭矩和转速方向相同
    direction = torque * motor_v
    # ========正转功率计算================
    # 6.37额定扭矩
    torque_p = np.where(abs_torque >= 6.37,
                        -(abs_torque - 6.37) / 27.13 + 1 + 0.3,
                        abs_torque / 6.37 + 0.3
                        )
    torque_p = np.minimum(np.maximum(torque_p, 0.01), 1)
    torque_p = torque_p ** 0.3

    # 19额定转速
    v_p = np.where(abs_motor_v >= 19,
                   -(np.minimum(np.maximum(abs_motor_v, 0), 25) - 19) / 6 + 1,
                   abs_motor_v / 19
                   )
    v_p = np.minimum(np.maximum(v_p, 0.01), 1)
    v_p = v_p ** 0.7

    efficient = (torque_p * v_p) * 0.95
    efficient = np.minimum(np.maximum(efficient, 0.01), 1)

    # ========反转功率计算================
    # 电机扭矩和转速方向相反的功率, 不考虑负功率, 反转时是一条斜线
    high = abs_torque - 33.5 / 21 * abs_motor_v
    # 电机扭矩和转速方向相同的功率
    power = np.where(direction >= 0,
                     (abs_torque * np.maximum(abs_motor_v, 0.1)) / efficient,
                     np.where(high >= 0,
                              np.abs(high) / np.sqrt(np.square(33.5 / 21) + 1) * 9.5,
                              0)
                     )
    return power, 0, efficient


def fig_contact(contacts, invalid_contact_num, co=1):
    # invalid_contact_num += len(contacts)
    if len(contacts) > 0:
        invalid_contact_num += 1 * co
    return invalid_contact_num


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
    )).reshape(batch, 3, 3)
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
    )).reshape(batch, 3, 3)
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
    )).reshape(batch, 3, 3)

    return m_z


@njit(cache=True)
def matrix_x_y_z(euler):
    batch = euler.shape[0]
    cos = np.cos(euler)
    sin = np.sin(euler)

    cos_x = cos[:, 0]
    sin_x = sin[:, 0]

    cos_y = cos[:, 1]
    sin_y = sin[:, 1]

    cos_z = cos[:, 2]
    sin_z = sin[:, 2]

    zeros = np.zeros_like(cos_z)
    ones = np.ones_like(cos_z)

    m_z = np.concatenate((
        cos_z, -sin_z, zeros,
        sin_z, cos_z, zeros,
        zeros, zeros, ones,
    )).reshape(batch, 3, 3)

    m_y = np.concatenate((
        cos_y, zeros, sin_y,
        zeros, ones, zeros,
        -sin_y, zeros, cos_y,
    )).reshape(batch, 3, 3)

    m_x = np.concatenate((
        ones, zeros, zeros,
        zeros, cos_x, -sin_x,
        zeros, sin_x, cos_x,
    )).reshape(batch, 3, 3)

    return m_z, m_y, m_x
