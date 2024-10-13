import numpy as np


'''
AB/AB髋关节，hip关节,knee关节, F前，B后，R右，L左
'''
motor_name = ['FRA', 'FRH', 'FRK', 'FLA', 'FLH', 'FLK', 'BRA', 'BRH', 'BRK', 'BLA', 'BLH', 'BLK']
total_num_joints = 25
# 电机下达指令的上下限

motor_num = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

# 注意这里1.1,本来是可以最大到1.3,但是这样可能学会双腿交叉，不太好看，于是改成1.1
motor_high =   [1.1,  2.9,  0.2,     1.3,  2.9,  0.2,     1.1,  1.0,  2.4,  1.3,  1.0,  2.4]
motor_high_1 = [0.8,  2.7,  -0.174,  1.0,  2.7,  -0.174,  0.8,  0.5,  2.25,  1.0,  0.5,  2.25]
motor_high_1 = np.array(motor_high_1)

motor_low =   [-1.3, -1.0, -2.4, -1.1, -1.0, -2.4, -1.3, -2.9,  -0.2,  -1.1, -2.9,  -0.2]
motor_low_1 = [-1.0, -0.5, -2.25, -0.8, -0.5, -2.25, -1.0, -2.7,  0.174, -0.8, -2.7,  0.174]
motor_low_1 = np.array(motor_low_1)
# 用来限制腿摆动角度


assert len(motor_name) == len(motor_num) == len(motor_high) == len(motor_low)

foot_num = [3, 7, 11, 15]
imu_num = 16
num_of_motor = len(motor_name)  # 电机数量

# 机体质量分布球
point_num = [17, 18, 19, 20, 21, 22, 23, 24]
num_of_point = len(point_num)

# 机体
BASE_LINK_ID = -1

# 动作和观测空间
action_dim = num_of_motor  # PD控制器的P
# 机体位置，机体角速度，旋转矩阵, 机体速度，各个关节速度\位置, 上一次的动作, 脚接触传感器, 重力, 机体质量, Friction
teacher_observation_dim = 3 + 3 + 9 + 3 + 2 * num_of_motor + action_dim + 4 + 3 + 1 + 1
student_observation_dim = 2 + 3 + 9 + 2 * num_of_motor + action_dim + 3

des_p_high = np.array(motor_high)  # 不同关节的位置上限
des_p_low = np.array(motor_low)

# kp_high = 80 * np.ones(num_of_motor)  # kp的上限
# kp_low = 40 * np.ones(num_of_motor)  # kp的下限
#
# feedforward_t_high = 10 * np.ones(num_of_motor)  # 前馈扭矩上限
# feedforward_t_low = -10 * np.ones(num_of_motor)

Random_coefficient = {  # 各种随机系数
    'global_Scaling': 0.03,  # 全局缩放
    'base_mass': 5,  # 机体最大增加的质量
    'mass': 0.15,  # 其他质量(%)
    'foot_Friction_high': 1.25,  # 脚摩擦系数
    'foot_Friction_low': 0.4,  # 脚摩擦系数
    'gravity': 0.03,  # 总重力随机(g)
    'gravity_angle_y': np.pi / 180 * 12,   # 最大12度斜坡
    'motor_torque': 0.03,  # 电机输出扭矩误差(%)
    # 'motor_torque_bias': 0.03,  # 电机输出扭矩bias(%)
    'motor_p_bias': 0.08,   # 电机位置的偏差，鼓励学习robust的策略
    'external_force': 10,  # 外部作用力+-(N)
}

init_Random_coefficient = {  # 各种随机系数
    'global_Scaling': 1e-16,  # 全局缩放
    'base_mass': 5,  # 机体最大增加的质量
    'mass': 0.15,  # 其他质量(%)
    'foot_Friction_high': 1.25,  # 脚摩擦系数
    'foot_Friction_low': 0.4,  # 脚摩擦系数
    'gravity': 0.03,  # 总重力随机(g)
    'gravity_angle_y': np.pi / 180 * 0.1,   # 最大12度斜坡
    'motor_torque': 0.03,  # 电机输出扭矩误差(%)
    # 'motor_torque_bias': 0.03,  # 电机输出扭矩bias(%)
    'motor_p_bias': 0.08,   # 电机位置的偏差，鼓励学习robust的策略
    'external_force': 0.1,  # 外部作用力+-(N)
}


# init_power = {
#     # 'motor_v_cost_weight': 0,
#     # 'power_cost_weight': 0,
#     'motor_v_acc_cost_weight': 0,
#     'lr': 3e-4,
#     # 'direction_weight': 1,
#     # 'invaled_work_state_cost': 0,
# }
#
#
# end_power = {
#     # 'motor_v_cost_weight': 2.0,
#     # 'power_cost_weight': -0.0005,
#     'motor_v_acc_cost_weight': 1,
#     'lr': 3e-4,
#     # 'direction_weight': 0,
#     # 'invaled_work_state_cost': 1.0,
# }

max_v = 3.5
min_v = 0.8
