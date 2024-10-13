import time
import gym
import numpy as np
from gym.utils import seeding
from numpy import ndarray
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
import os
from Env.config import *
from Env.Env_random import Env_random
import torch
from Env.functions import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix
from Env.utlis import fig_des_p, cal_m_power, torque_limit, power_limit, motor
from scipy.spatial.transform import Rotation

path = os.getcwd()  # 获取当前工作目录路径
print(path)

one_vector = torch.tensor([0, 0, -1], dtype=torch.double)

# =================check==============
euler = np.random.uniform(-np.pi, np.pi, (3,))
q1 = np.array(pybullet.getQuaternionFromEuler(euler))
q2 = Rotation.from_euler('xyz', euler).as_quat()

if not (np.abs((q1 - q2)) < 1e-14).all():
    print(q1, q2)
assert (np.abs((q1 - q2)) < 1e-14).all() == True

# e1 = np.array(pybullet.getEulerFromQuaternion(q1))
# e2 = Rotation.from_quat(q1).as_euler('xyz')
#
# if not (np.abs((e1 - e2)) < 1e-14).all():
#     print(e1, e2)
# assert (np.abs((e1 - e2)) < 1e-14).all() == True

m1 = np.array(pybullet.getMatrixFromQuaternion(q1)).reshape(3, 3)
m2 = Rotation.from_quat(q1).as_matrix()

if not (np.abs((m1 - m2)) < 1e-14).all():
    print(m1, m2)
assert (np.abs((m1 - m2)) < 1e-14).all() == True

# =============================


class Minicheeth(gym.Env):
    """ 用于模拟两个timestep之间机器人的运动 """

    def __init__(
            self,
            # 获取默认储存路径
            urdf_root: str = path + '/mini_cheetah/mini_cheetah.urdf',
            motor_torque_limit: float = 33.5,  # 电机扭矩限制
            motor_overheat_torque: float = 25,  # 过热停机转矩
            motor_overheat_protection: bool = False,  # 是否长时间（过热停机时间）关闭施加大转矩（过热停机转矩）的电机
            motor_overheat_time: float = 2.,  # 过热停机时间
            GUI: bool = False,  # 是否渲染
            motor_kd: float = 0.5,
            pd_timestep: float = 0.001,  # PD控制器的timestep
            use_randomize: bool = True,  # 是否对环境随机绕动
            use_default_terrain: bool = True,  # 用哪种地形
            set_time: float = 0,  # 开始前的放置时间
            x_y_random_posture: float = np.pi,  # xy随机姿态
            start_z: float = 0.48,  # 初始化Z的高度
            heightPerturbationRange: float = 0.025,  # 复杂地形的随即地形高度
            init_position_high=18,
            init_position_low=14,
            seed=None,
            max_t=33.5,
    ) -> None:
        self.seed(seed)

        self._time_step = 0.001  # pybullet的默认time step
        self.urdf_root = urdf_root

        self.motor_torque_limit = motor_torque_limit
        self.motor_overheat_torque = motor_overheat_torque
        self.motor_overheat_protection = motor_overheat_protection
        self.motor_overheat_time = motor_overheat_time

        self.motor_kd = motor_kd
        self.pd_timestep = pd_timestep
        self.pd_freq = int(pd_timestep / self._time_step)

        self.total_it = 0  # pybullet timestep次数

        self.dafault_root = pybullet_data.getDataPath(),  # 获取pybullet默认储存路径

        self.over_heat_num = np.zeros(num_of_motor)  # 电机过热计数器
        self.over_heat_num2 = np.zeros(num_of_motor)  # 过热恢复计数器

        self.is_render = GUI
        if self.is_render:  # 如果渲染
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self.use_randomize = use_randomize
        self.use_default_terrain = use_default_terrain
        self.terrain = None
        self.quadruped = None
        self.terrainShape = None
        self.set_time = set_time
        self.x_y_random_posture = x_y_random_posture
        self.start_z = start_z

        self._pybullet_client.setRealTimeSimulation(False)  # 不使用实时计算

        # 外部作用力
        self.external_pos = [0, 0, 0]
        self.heightPerturbationRange = heightPerturbationRange

        # 随机初始化时的位置
        self.init_position_high, self.init_position_low = init_position_high, init_position_low

        # 初始化
        self.init()

        # 创建随机类
        self.Env_random = Env_random()
        self.all_random_message = self.Env_random.get_all_random()

        self.delay_p = [np.zeros(12) for x in range(5)]  # 用来模拟延迟
        self.delay_v = [np.zeros(12) for x in range(5)]  # 用来模拟延迟

        # gui debug用
        self.last_base_p_1 = [0, 0, 0]
        self.last_foot_p_1 = [[0, 0, 0] for _ in range(4)]

        self.motor_torque = np.zeros(num_of_motor)
        self.true_torque = np.zeros(num_of_motor)
        # self.reset_robot_state()

        self.update_all = True

    def reset_robot_state(self):
        self.robot_state = {
            'base_v_world': np.zeros(3),
            'base_v': np.zeros(3),
            'last_base_v_world': np.zeros(3),
            'imu_w': np.zeros(3),
            'imu_R': np.zeros(9),
            'imu_v': np.zeros(3),
            'last_imu_v': np.zeros(3),
            'base_w': np.zeros(3),
            'base_w_world': np.zeros(3),
            # 'base_o': np.zeros(3),
            'base_R': np.zeros(9),
            'origin_R': np.zeros(9),
            'base_p': np.zeros(3),
            'foot_contact': np.zeros(4),
            'invalid_contact_num': 0,
            'origin_base_euler': np.zeros(3),
            'motor_torque': np.zeros(num_of_motor),
            'true_torque': np.zeros(num_of_motor),
            'motor_v': np.zeros(num_of_motor),
            'last_motor_v': np.zeros(num_of_motor),
            'motor_p': np.zeros(num_of_motor),
            'foot_p': np.zeros(4),
            'foot_v': np.zeros(4),
            'power': 0,
            'point_v': np.zeros((num_of_point, 3)),
            'last_point_v': np.zeros((num_of_point, 3)),
        }

    def seed(self, seed=None) -> list:  # 设置种子
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def init(self):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setTimeStep(self._time_step)
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)

        self.create_terrain(default=self.use_default_terrain)  # 构建地形

    def reset_(self) -> None:  # 重置机器人
        self.reset_robot_state()
        self.motor_torque = np.zeros(num_of_motor)
        self.total_it = 0  # pybullet timestep次数
        self.over_heat_num = np.zeros(num_of_motor)  # 电机过热计数器
        self.over_heat_num2 = np.zeros(num_of_motor)  # 过热恢复计数器
        self.delay_p = [np.zeros(12) for x in range(5)]  # 用来模拟延迟
        self.delay_v = [np.zeros(12) for x in range(5)]  # 用来模拟延迟

        # 坠落姿态随机
        if self.use_randomize:
            # 位置随机
            distance = self.np_random.uniform(self.init_position_low, self.init_position_high)
            angle = self.np_random.uniform(-np.pi, np.pi)
            x = distance * np.sin(angle)
            y = distance * np.cos(angle)
            start_Pos = [x, y, self.start_z]  # 初始化随机位置,从0.5m的地方坠落

            # 坠落姿态随机
            # random_Euler = list(self.np_random.uniform(-np.pi, np.pi, size=3))
            random_Euler = [float(self.np_random.uniform(-self.x_y_random_posture, self.x_y_random_posture)),
                            float(self.np_random.uniform(-self.x_y_random_posture, self.x_y_random_posture)),
                            float(self.np_random.uniform(-np.pi, np.pi))
                            ]
            self.Env_random.update_global_Scaling(self.np_random)
            self.all_random_message = self.Env_random.get_all_random()
        else:
            random_Euler = [0, 0, 0]
            start_Pos = [1, 1, self.start_z]
        startOrientation = self._pybullet_client.getQuaternionFromEuler(random_Euler)

        # 设置接触
        urdfFlags = pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        # urdfFlags = pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        if self.quadruped is None:
            self.quadruped = self._pybullet_client.loadURDF(self.urdf_root,
                                                            useMaximalCoordinates=False,
                                                            basePosition=start_Pos,
                                                            baseOrientation=startOrientation,
                                                            flags=urdfFlags,  # 自身碰撞,但是去除关节间碰撞
                                                            # 缩放机器人尺寸大小
                                                            globalScaling=self.all_random_message['global_Scaling'],
                                                            # 全局缩放系数
                                                            )

            # lower collision
            lower_lags = [2, 3, 6, 7, 10, 11, 14, 15]
            for l in lower_lags:
                self._pybullet_client.setCollisionFilterPair(self.quadruped, self.quadruped, -1, l, enableCollision=1)
                for l1 in lower_lags:
                    if l1 > (l + 1):
                        self._pybullet_client.setCollisionFilterPair(self.quadruped, self.quadruped, l, l1,
                                                                     enableCollision=1)

            # Joint的数量
            numJoints = self._pybullet_client.getNumJoints(self.quadruped)
            assert numJoints == total_num_joints

            # 设置颜色
            self._pybullet_client.changeVisualShape(self.quadruped, -1, rgbaColor=[1, 1, 1, 1])
            for j in range(total_num_joints):
                self._pybullet_client.changeVisualShape(self.quadruped, j, rgbaColor=[1, 1, 1, 1])
            self._RecordInfoFromURDF()  # 读取各种信息(重量，摩擦系数)

            if self.use_randomize:
                self.Env_random.reset_random_message(self.np_random, self.base_mass, self.leg_mass)
                self.Env_random.apply(self._pybullet_client, self.quadruped, self.terrain)
                self.all_random_message = self.Env_random.get_all_random()
            else:
                self.set_all()

        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, start_Pos, startOrientation)

        for j in motor_num:
            self._pybullet_client.resetJointState(self.quadruped, j, 0, 0)  # 设置关节角归位
            self._pybullet_client.setJointMotorControl2(self.quadruped, j, pybullet.POSITION_CONTROL,
                                                        targetPosition=0,
                                                        force=0)
        # 把机器人先放2秒
        step = int(self.set_time / self._time_step)
        for x in range(step):
            self.stepSimulation_with_external()
        self.update_all = True
        self.flash_robot_state()
        return

    def close(self) -> None:
        self._pybullet_client.disconnect()

    def gui_show(self) -> None:  # time.sleep()和调整相机位置
        base_pos = self.robot_state['base_p']
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]  # 相机的距离，没什么用
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [
            0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
            curTargetPos[2]
        ]
        # 调整相机位置
        self._pybullet_client.resetDebugVisualizerCamera(1.,  # 距离
                                                         yaw,
                                                         pitch,
                                                         base_pos)  # 位置

        # 添加一些运动曲线

        # if self.total_it % 15 == 0:
        #     # 机体位置
        #     base_p, base_o = self._pybullet_client.getBasePositionAndOrientation(bodyUniqueId=self.quadruped, )
        #     base_p = list(base_p)
        #     if self.last_base_p_1 == [0, 0, 0]:
        #         self._pybullet_client.addUserDebugLine(base_p, base_p, lineColorRGB=[1, 0, 0], lineWidth=3,
        #                                                lifeTime=10)
        #     else:
        #         self._pybullet_client.addUserDebugLine(self.last_base_p_1, base_p, lineColorRGB=[1, 0, 0], lineWidth=3,
        #                                                lifeTime=10)
        #     self.last_base_p_1 = base_p
        #     n = 0
        #     foot_color = [[0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]]
        #     for x in foot_num:
        #         foot_p = np.array(self._pybullet_client.getLinkState(self.quadruped, x)[0])
        #         foot_p = list(foot_p)
        #         if self.last_foot_p_1[n] == [0, 0, 0]:
        #             self._pybullet_client.addUserDebugLine(foot_p, foot_p, lineColorRGB=foot_color[n], lineWidth=3,
        #                                                    lifeTime=10)
        #         else:
        #             self._pybullet_client.addUserDebugLine(self.last_foot_p_1[n], foot_p, lineColorRGB=foot_color[n], lineWidth=3,
        #                                                    lifeTime=10)
        #         self.last_foot_p_1[n] = foot_p
        #         n += 1

        # 然后是脚的位置

        # time.sleep(self._time_step * 0.9)

    def get_imu_v(self):  # 机体坐标的IMU数据
        imu_State = self._pybullet_client.getLinkState(self.quadruped, imu_num, computeLinkVelocity=1)
        origin_imu_R = np.array(self._pybullet_client.getMatrixFromQuaternion(imu_State[1])).reshape(3, 3)
        imu_v_world = np.array(imu_State[-2]).reshape(3, 1)
        imu_v = np.matmul(origin_imu_R.T, imu_v_world).reshape(-1)
        imu_w_world = np.array(imu_State[-1]).reshape(3, 1)
        imu_w = (origin_imu_R.T @ imu_w_world).reshape(-1)
        return imu_v, imu_State, origin_imu_R.reshape(-1), imu_w, imu_w_world

    def get_point_v(self):
        point_state = self._pybullet_client.getLinkStates(self.quadruped, point_num, computeLinkVelocity=1)
        v = []
        for state in point_state:
            v.append(state[-2])
        return np.array(v)

    def PD_control(self, kp, des_p, des_v, feedforward_t) -> None:
        current_p = self.robot_state['motor_p']
        current_v = self.robot_state['motor_v']
        true_motor_v = self.robot_state['true_motor_v']

        if self.total_it % self.pd_freq == 0:
            torque = motor(kp, des_p, current_p, self.motor_kd, des_v, current_v, feedforward_t, true_motor_v)
            self.true_torque = torque.copy()
            torque = torque * self.np_random.uniform(0.975, 1.025, size=torque.shape)
            # self.true_torque = torque.copy()
            # 电机启动转矩
            self.motor_torque = torque - np.sign(true_motor_v) * self.all_random_message['motor_start_torque']

        self.applying_torque(self.motor_torque)

        self.total_it += 1
        self.flash_robot_state()

    # 把扭矩应用到指定的电机上
    def applying_torque(self,
                        torque: ndarray,  # np.array
                        # motor,      # list
                        ) -> None:
        # self.motor_torque = torque
        # 电机执行的扭矩会有误差
        if self.motor_overheat_protection:  # 如果电机会过热
            # index = motor_num.index(motor)
            over_heat_mask = np.array(self.over_heat_num2 < (self.motor_overheat_time / self._time_step),
                                      dtype='int32')  # 如果停转超过1秒，重置计数器
            self.over_heat_num = self.over_heat_num * over_heat_mask
            self.over_heat_num2 = self.over_heat_num2 * over_heat_mask

            # if self.over_heat_num2[index] > (1 / self._time_step):  # 如果停转超过1秒，重置计数器
            #     self.over_heat_num[index] = 0
            #     self.over_heat_num2[index] = 0

            over_heat_mask = np.array(self.over_heat_num < (self.motor_overheat_time / self._time_step),
                                      dtype='int32')  # 如果过热超过1秒
            torque = torque * over_heat_mask  # 停转1秒
            self.over_heat_num2 += (1 - over_heat_mask)  # 停转计数器+1

            # if self.over_heat_num[index] > (1 / self._time_step):
            #     # 如果过热超过1秒
            #     torque = 0
            #     # 电机停转1秒
            #     self.over_heat_num2[index] += 1

            over_torque_mask = np.array(np.abs(torque) > self.motor_overheat_torque, dtype='int32')
            self.over_heat_num += (over_torque_mask * over_heat_mask)

            # elif torque > self.motor_overheat_torque:
            #     self.over_heat_num[index] += 1

            self.over_heat_num += ((over_torque_mask - 1) * over_heat_mask)
            self.over_heat_num = self.over_heat_num.clip(0, 1e5)
            # print(self.over_heat_num)
            # else:
            #     self.over_heat_num[index] -= 1
            #     self.over_heat_num[index] = np.clip(self.over_heat_num[index], 0, 1e5)
        self._pybullet_client.setJointMotorControlArray(self.quadruped,
                                                        motor_num,
                                                        controlMode=pybullet.TORQUE_CONTROL,
                                                        forces=torque)
        self.stepSimulation_with_external()
        return torque

    def flash_robot_state(self):

        last_base_v_world = self.robot_state['base_v_world']
        last_motor_torque = self.robot_state['motor_torque']
        last_motor_v = self.robot_state['motor_v']

        # 机体世界坐标系速度和角速度
        base_v_world, base_w_world = self._pybullet_client.getBaseVelocity(bodyUniqueId=self.quadruped, )

        # 机体位置和四元数
        base_p, base_o = self._pybullet_client.getBasePositionAndOrientation(bodyUniqueId=self.quadruped, )
        base_p = np.array(base_p)
        origin_R = self._pybullet_client.getMatrixFromQuaternion(base_o)

        # from scipy.spatial.transform import Rotation
        # print(origin_R - Rotation.from_quat(base_o).as_matrix().flatten())

        origin_R = np.array(origin_R).reshape(3, 3)

        # 先处理速度
        # TODO 用机体坐标系还是世界坐标系??
        base_v_world = np.array(base_v_world).reshape(3, 1)
        base_v = np.matmul(origin_R.T, base_v_world)  # 机体坐标系的线速度

        # 电机位置，速度，扭矩
        joint_states = self._pybullet_client.getJointStates(self.quadruped, motor_num)
        motor_p = []
        motor_v = []
        for state in joint_states:
            motor_p.append(state[0])
            motor_v.append(state[1])
        true_motor_p = np.array(motor_p)
        true_motor_v = np.array(motor_v)
        motor_p = true_motor_p + self.all_random_message['motor_p_bias'] + self.np_random.uniform(-0.02, 0.02,
                                                                                                  size=(num_of_motor,))
        motor_v = true_motor_v * self.np_random.uniform(0.96, 1.04, size=(num_of_motor,))

        power = cal_m_power(self.true_torque, true_motor_v)

        foot_contact, foot_p, foot_v = self.contact_detect()
        foot_contact = np.array(foot_contact)

        last_point_v = self.robot_state['point_v']
        point_v = self.get_point_v()

        if self.update_all:
            last_imu_v = self.robot_state['imu_v']
            imu_v, imu_State, origin_imu_R, imu_w, imu_w_world = self.get_imu_v()

            invalid_contact_num = self.invalid_contact_detect()
            base_w_world = np.array(base_w_world)
            base_w = np.matmul(origin_R.T, base_w_world.reshape(3, 1)).flatten()  # 机体坐标系的角速度

            origin_base_euler = np.array(self._pybullet_client.getEulerFromQuaternion(base_o)).flatten()

            # ========旧版算法==============
            # g = torch.tensor([self.all_random_message['gravity_x'],
            #                   self.all_random_message['gravity_y'],
            #                   self.all_random_message['gravity_z'], ], dtype=torch.double)
            #
            # g_norm = g / g.norm()
            # g_axis_ = torch.cross(g_norm, one_vector, -1)  # 叉乘
            # g_axis_ = g_axis_ / (g_axis_.norm() + 1e-17)  # 转成单位向量
            # # cos = (g_norm.mul(one_vector)).sum()
            # # angle2 = torch.arccos(cos).reshape(1, )
            # angle = torch.tensor([abs(float(self.all_random_message['gravity_euler'][1]))],
            #                      dtype=torch.double)  # 两个角度是一样的
            # # print(angle2 - angle)
            #
            # mmm_ = axis_angle_to_matrix(g_axis_, angle)
            # =========下面是新方法

            mmm_ = self.all_random_message['m2'].T
            # print(self.all_random_message['m2'] @ mmm_)

            # origin_R_ = torch.tensor(origin_R, dtype=torch.double).reshape(3, 3)
            base_R = (mmm_ @ origin_R.reshape(3, 3)).reshape(-1)
            imu_R = (mmm_ @ origin_imu_R.reshape(3, 3)).reshape(-1)

            # print(mmm_ @ g_norm - one_vector)

            # 重力偏移后的IMU欧拉角
            # base_euler = matrix_to_euler_angles(base_R, 'ZYX').flip(-1).numpy()

            # base_o = np.array(base_o)
            # print(base_R - imu_R)
        else:
            base_R = None
            origin_imu_R = None
            invalid_contact_num = None
            origin_base_euler = None
            base_w_world = self.robot_state['base_w_world']
            base_w = self.robot_state['base_w']
            imu_v, imu_R, imu_w = self.robot_state['imu_v'], self.robot_state['imu_R'], self.robot_state['imu_w']
            last_imu_v = self.robot_state['last_imu_v']

        self.robot_state = {
            'base_v_world': base_v_world.flatten(),
            'base_v': base_v.flatten(),
            'last_base_v_world': last_base_v_world,
            'imu_w': imu_w,
            'origin_imu_R': origin_imu_R,
            'imu_R': imu_R,
            'imu_v': imu_v,
            'last_imu_v': last_imu_v,
            'base_w': base_w,
            'base_w_world': base_w_world,
            # 'base_o': base_o,
            'base_R': base_R,
            'origin_R': origin_R.flatten(),
            'base_p': base_p.flatten(),
            'foot_contact': foot_contact,
            'invalid_contact_num': invalid_contact_num,
            'origin_base_euler': origin_base_euler,
            'motor_torque': self.motor_torque.copy(),
            'last_motor_torque': last_motor_torque,
            'true_torque': self.true_torque.copy(),
            'motor_v': motor_v,
            'last_motor_v': last_motor_v,
            'motor_p': motor_p,
            'true_motor_p': true_motor_p,
            'true_motor_v': true_motor_v,
            'foot_p': foot_p,
            'foot_v': foot_v,
            'power': power,
            'point_v': point_v,
            'last_point_v': last_point_v,
        }

    def set_all(self) -> None:  # 没有任何随机的设置
        # 重力
        self._pybullet_client.setGravity(0, 0, -9.815)  # 设置重力

    def _RecordInfoFromURDF(self) -> None:  # 读取重量
        # 先读取机体
        self.base_mass, self.leg_mass, self.point_mass, self.total_mass = self.get_current_mass()

        # print('总质量', self.base_mass + np.array(self.leg_mass).sum())
        # print(self.base_mass, self.leg_mass)

        # # 电机摩擦系数
        # self.motor_friction = []
        # for x in motor_num:
        #     self.motor_friction.append(self._pybullet_client.getDynamicsInfo(self.quadruped, x)[1])
        #
        # # 脚摩擦系数
        # self.foot_friction = []
        # for x in foot_num:
        #     self.foot_friction.append(self._pybullet_client.getDynamicsInfo(self.quadruped, x)[1])

    def get_current_mass(self):
        base_mass = self._pybullet_client.getDynamicsInfo(self.quadruped, BASE_LINK_ID)[0]
        # 然后是各个link的重量
        leg_mass = []
        for x in motor_num:
            leg_mass.append(self._pybullet_client.getDynamicsInfo(self.quadruped, x)[0])

        point_mass = []
        for x in point_num:
            point_mass.append(self._pybullet_client.getDynamicsInfo(self.quadruped, x)[0])

        foot_mass = []
        for x in foot_num:
            foot_mass.append(self._pybullet_client.getDynamicsInfo(self.quadruped, x)[0])
        foot_mass.append(self._pybullet_client.getDynamicsInfo(self.quadruped, imu_num)[0])

        total_mass = base_mass + sum(leg_mass) + sum(point_mass) + sum(foot_mass)

        return base_mass, leg_mass, point_mass, total_mass

    def get_joint_information(self) -> None:
        for x in motor_num:
            l = self._pybullet_client.getJointInfo(self.quadruped, x)
            print(
                x, '关节阻尼:', l[6], '\n',
                x, '关节摩擦:', l[7], '\n',
                x, '关节限制低:', l[8], '\n',
                x, '关节限制高:', l[9], '\n',
                x, '最大力:', l[10], '\n',
                x, '最大速度:', l[11], '\n',
                x, '名字:', l[12], '\n',
            )

    def get_information(self) -> None:
        x = -1
        l = self._pybullet_client.getDynamicsInfo(self.quadruped, x)
        print(x, '质量:', l[0], '\n',
              x, '摩擦系数:', l[1], '\n',
              x, '惯性矩:', l[2], '\n',
              x, '惯性位置:', l[3], '\n',
              x, '惯性方向:', l[4], '\n',
              x, '恢复系数:', l[5], '\n',
              x, '滚动摩擦:', l[6], '\n',
              x, '旋转摩擦:', l[7], '\n',
              x, '接触阻尼:', l[8], '\n',
              x, '接触刚性:', l[9], '\n',
              x, '刚体类型:', l[10], '\n',
              )
        for x in range(total_num_joints):
            l = self._pybullet_client.getDynamicsInfo(self.quadruped, x)
            print(x, '质量:', l[0], '\n',
                  x, '摩擦系数:', l[1], '\n',
                  x, '惯性矩:', l[2], '\n',
                  x, '惯性位置:', l[3], '\n',
                  x, '恢复系数:', l[5], '\n',
                  x, '滚动摩擦:', l[6], '\n',
                  x, '旋转摩擦:', l[7], '\n',
                  x, '接触阻尼:', l[8], '\n',
                  x, '接触刚性:', l[9], '\n',
                  x, '刚体类型:', l[10], '\n',
                  )

    def stepSimulation_with_external(self):
        self._pybullet_client.applyExternalForce(objectUniqueId=self.quadruped,
                                                 linkIndex=-1,
                                                 forceObj=self.all_random_message['external_force'],
                                                 posObj=self.all_random_message['external_pos'],
                                                 flags=pybullet.LINK_FRAME
                                                 )
        self._pybullet_client.stepSimulation()
        if self.is_render:
            self.gui_show()

    def create_terrain(self, default=False, ):  # 地形生成
        if default:
            self.terrain = self._pybullet_client.loadURDF("%s/plane.urdf" % self.dafault_root)
            self._pybullet_client.changeVisualShape(self.terrain, -1,
                                                    rgbaColor=[1, 1, 1, 0.9])  # 用changeVisualShape更改形状的纹理
            # # 设置地面的摩擦系数
            # self._pybullet_client.changeDynamics(self.terrain, -1,
            #                                      lateralFriction=float(
            #                                          self.np_random.uniform(Random_coefficient['plane_Friction_low'],
            #                                                                 Random_coefficient['plane_Friction_high'])))
        else:
            numHeightfieldRows = 512
            numHeightfieldColumns = 512
            heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
            for j in range(int(numHeightfieldColumns / 2)):
                for i in range(int(numHeightfieldRows / 2)):
                    height = self.np_random.uniform(-self.heightPerturbationRange, self.heightPerturbationRange)
                    heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
                    heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

            meshScale = np.clip(self.heightPerturbationRange / 0.15, 0.1, 0.2)
            terrainShape = self._pybullet_client.createCollisionShape(
                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[meshScale, meshScale, 1],
                heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
                heightfieldData=heightfieldData,
                numHeightfieldRows=numHeightfieldRows,
                numHeightfieldColumns=numHeightfieldColumns)
            self.terrain = self._pybullet_client.createMultiBody(0, terrainShape)
            self._pybullet_client.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1])
            self._pybullet_client.changeVisualShape(self.terrain, -1, rgbaColor=[0.5, 0.5, 0.5, 0.9])

    def invalid_contact_detect(self):
        # 腿非法接触检测
        leg_num = [[0, 1, 2, 3],  # 4条腿
                   [4, 5, 6, 7],
                   [8, 9, 10, 11],
                   [12, 13, 14, 15]]
        num = 0
        for leg in range(len(leg_num) - 1):  # 腿之间的碰撞
            for motor in leg_num[leg]:
                # if motor not in [2, 3, 6, 7, 10, 11, 14, 15]:
                #     leg_num_ = np.array(leg_num)[num + 1:, :].flatten().tolist()
                # else:
                #     leg_num_ = np.array(leg_num)[num + 1:, :2].flatten().tolist()
                leg_num_ = np.array(leg_num)[num + 1:, :].flatten().tolist()
                for other_motor in leg_num_:
                    contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.quadruped,
                                                                      linkIndexA=motor, linkIndexB=other_motor)
                    if len(contacts) > 0:
                        return 1
            num += 1

        # foot
        # for foot in foot_num:  # 腿之间的碰撞
        #     for other_foot in foot_num:
        #         if foot != other_foot:
        #             contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.quadruped,
        #                                                               linkIndexA=foot, linkIndexB=other_foot)
        #             if len(contacts) > 0:
        #                 return 1

        for joint in range(16):  # 检查机体的碰撞
            if joint not in [0, 4, 8, 12]:
                contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.quadruped,
                                                                  linkIndexA=BASE_LINK_ID, linkIndexB=joint)
                if len(contacts) > 0:
                    return 1

        # 脚非法接触检测
        for n in range(4):
            contacts = self._pybullet_client.getContactPoints(bodyA=self.terrain, bodyB=self.quadruped,
                                                              linkIndexB=foot_num[n] - 1)
            for contact in contacts:  # 检查所有接触
                link_index = contact[4]
                if link_index == foot_num[n] - 1:
                    contact_p = np.array(contact[6])
                    # 计算和足的距离
                    # print(np.sqrt(((foot_p - contact_p) ** 2).sum()), np.sqrt(((foot_p - contact_p) ** 2)).sum())
                    d = np.sqrt(((self.robot_state['foot_p'][n] - contact_p) ** 2).sum())
                    if d > 0.05:  # 如果接触距离远，则判定为非法接触
                        return 1

        return 0

    def contact_detect(self):
        # invalid_contact_num = 0  # 统计非法接触
        valid_contact = [0 for _ in foot_num]  # 统计合法接触
        # n = 0  # 计算第几个脚

        foot_p_list = []
        foot_v_list = []

        foot_states = self._pybullet_client.getLinkStates(self.quadruped, foot_num, computeLinkVelocity=1)
        for state in foot_states:
            foot_p_list.append(state[0])
            foot_v_list.append(state[-2])
        # 脚接触检测
        n = 0
        for x in foot_num:
            contacts = self._pybullet_client.getContactPoints(bodyA=self.terrain, bodyB=self.quadruped,
                                                              linkIndexB=x)
            if len(contacts) > 0:
                f = 0
                for c in contacts:
                    f += c[-5]
                if f > 0:
                    valid_contact[n] = 1
            n += 1
        return valid_contact, np.array(foot_p_list), np.array(foot_v_list)


if __name__ == '__main__':
    M = Minicheeth(GUI=True, motor_kd=0.1, motor_overheat_protection=False, use_randomize=True,
                   use_default_terrain=False)
    M.reset_()
    #     # while True:
    #     #     time.sleep(0.5)
    #     #     M.reset_()
    #     # print(M.get_imformation())
    #     n = 0
    action = np.array([0, 0., -2.5, 0., 0., 0., 0.0, 0., 0., 0.0, 0., 0.])
    a = np.array([0., 0., 0., 0.0, 0., 0., 0.0, 0., 0., 0.0, 0., 0.])
    #     np.set_printoptions(suppress=True)
    #
    #     # 下面代码可以手动指定位置
    #     # l = 1
    #     # z = self.np_random.uniform(-np.pi, np.pi)
    #     # start_Pos = [-l * np.cos(z), -l * np.sin(z), 0.48]
    #     # Euler = [0, 0, z]
    #     # Orientation = M._pybullet_client.getQuaternionFromEuler(Euler)
    #     # M._pybullet_client.resetBasePositionAndOrientation(M.quadruped, start_Pos, Orientation)
    #     # time.sleep(1111)
    #
    #     # 计算跟踪误差
    #     error = []
    #     num = 0
    n = 0
    while True:
        # base_v, base_w, base_p, base_o, base_euler, origin_R, motor_p, motor_v, foot_contact, origin_base_euler = M.get_ob()
        if n % 15 == 0:
            # #   action = -action
            # error.append(abs(a[1] - motor_p[1]))
            # num += 1
            # if num > 20:
            #     print('===============', np.array(error).mean(), np.array(error).std(), a[1])
            #     break
            a = a * 0.9 + action * 0.1
        # a = action
        M.PD_control(30 * np.ones(num_of_motor), a,
                     np.zeros(num_of_motor), np.ones(num_of_motor) * -0)
        #         # print(n, np.round(M.motor_torque[1], 2), np.round(motor_v[1], 2), np.round(motor_p[1], 2))
        #         # print(base_p[-1])FF
        #         # M.contact_detect()
        print(M.robot_state['motor_p'])
        time.sleep(0.001)
        #         # M.reset_()
        n += 1
