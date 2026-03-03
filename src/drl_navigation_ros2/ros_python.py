import time
import rclpy
from ros_nodes import (
    ScanSubscriber,
    OdomSubscriber,
    ResetWorldClient,
    SetModelStateClient,
    CmdVelPublisher,
    MarkerPublisher,
    PhysicsClient,
    SensorSubscriber,
)
import numpy as np
from geometry_msgs.msg import Pose, Twist
from squaternion import Quaternion


# 仿真环境类
class ROS_env:
    def __init__(
        self,
        init_target_distance=2.0,   # 初始目标距离
        target_dist_increase=0.001, # 每次达到目标后目标距离增加的值
        max_target_dist=8.0,        # 目标距离的最大值
        target_reached_delta=0.5,   # 到达目标的距离阈值
        collision_delta=0.4,        # 发生碰撞的距离阈值
        args=None,
    ):
        rclpy.init(args=args)
        # 速度指令发布者
        self.cmd_vel_publisher = CmdVelPublisher()

        # 雷达传感器和定位订阅者，没用到，被sensor_subscriber替代了
        self.scan_subscriber = ScanSubscriber()
        self.odom_subscriber = OdomSubscriber()

        # 物体状态发布者，用于重置位置，名字取的不好，并非只用于重置机器人位置
        self.robot_state_publisher = SetModelStateClient()
        # 世界重置服务客户端
        self.world_reset = ResetWorldClient()
        # 物理客户端，用于控制仿真环境的暂停
        self.physics_client = PhysicsClient()

        # 目标点可视化发布者
        self.publish_target = MarkerPublisher()

        # 四个固定障碍物的坐标
        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]

        # 传感器数据订阅者，同时订阅雷达和定位数据
        self.sensor_subscriber = SensorSubscriber()

        # 记录初始目标距离、目标距离增加值、最大目标距离、到达目标的距离阈值、碰撞的距离阈值
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta

        # 记录目标点的坐标
        self.target = self.set_target_position([0.0, 0.0])

    # 进行一个step（一步）
    def step(self, lin_velocity=0.0, ang_velocity=0.1, last_distance=0.0):
        # 发布速度指令
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        # 启动世界
        self.physics_client.unpause_physics()
        # 手动控制持续时间
        time.sleep(0.1)
        # 订阅最新的传感器数据
        rclpy.spin_once(self.sensor_subscriber)
        # 暂停世界
        self.physics_client.pause_physics()

        # 返回最新的传感器数据
        (
            latest_scan,
            latest_position,
            latest_orientation,
            latest_vel
        ) = self.sensor_subscriber.get_latest_sensor()

        # 获得到目标的距离和朝向
        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )

        # 判断是否发生碰撞和是否到达目标
        collision = self.check_collision(latest_scan)
        goal = self.check_target(distance, collision)

        # 计算奖励
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, last_distance, distance)

        # 返回所有状态所需的数据、是否碰撞、是否到达、奖励
        return latest_scan, distance, cos, sin, collision, goal, action, reward, latest_vel

    # 重置环境
    def reset(self):
        # 重置世界，此时机器人仅是位置复原，可能还有速度
        self.world_reset.reset_world()
        # 发布0速度让机器人停下
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )

        # 重置障碍物位置记录（如源码设置一次环境后是8个随机障碍物，但此时需要变回4个）
        # 为后面重新设置随机障碍物等位置做准备
        self.element_positions = [
            [-2.93, 3.17],
            [2.86, -3.0],
            [-2.77, -0.96],
            [2.83, 2.93],
        ]
        # 重新设置四个随机障碍物的位置和机器人位置、目标位置
        self.set_positions()

        # 发布可视化目标点
        self.publish_target.publish(self.target[0], self.target[1])
        
        # 重置完先0速进行一个step以获取初始环境信息
        latest_scan, distance, cos, sin, _, _, action, reward, vel = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward, vel

    # 设置评估环境
    def eval(self, scenario):
        self.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)

        # 按生成的场景设置所有物体位置
        self.target = [scenario[-1].x, scenario[-1].y]
        self.publish_target.publish(self.target[0], self.target[1])

        for element in scenario[:-1]:
            self.set_position(element.name, element.x, element.y, element.angle)

        # 开局静止一秒钟让物体位置稳定，然后获取初始环境信息
        self.physics_client.unpause_physics()
        time.sleep(1)
        latest_scan, distance, cos, sin, _, _, a, reward, vel = self.step(
            lin_velocity=0.0, ang_velocity=0.0
        )
        return latest_scan, distance, cos, sin, False, False, a, reward, vel


    # 设置物体位置，最后发布
    def set_position(self, name, x, y, angle):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        self.robot_state_publisher.set_state(name, pose)
        rclpy.spin_once(self.robot_state_publisher)

    # 设置随机障碍物位置
    def set_random_position(self, name):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        # 每设置完一个随机障碍物就添加到障碍物序列中
        self.element_positions.append([x, y])
        self.set_position(name, x, y, angle)

    # 检查到已有障碍物的距离是否大于阈值
    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

    # 设置目标位置
    def set_target_position(self, robot_position):
        pos = False
        while not pos:
            x = np.clip(
                robot_position[0]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -4.0,
                4.0,
            )
            y = np.clip(
                robot_position[1]
                + np.random.uniform(-self.target_dist, self.target_dist),
                -4.0,
                4.0,
            )
            pos = self.check_position(x, y, 1.2)
        self.element_positions.append([x, y])
        return [x, y]

    # 设置机器人位置
    def set_robot_position(self):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.set_position("turtlebot3_waffle", x, y, angle)
        return x, y

    # 设置随机障碍物位置、机器人位置、目标位置
    def set_positions(self):
        for i in range(4, 8):
            name = "obstacle" + str(i + 1)
            self.set_random_position(name)

        # 设置机器人位置，考虑到所有固定和随机障碍物
        robot_position = self.set_robot_position()

        # 设置目标位置，考虑到机器人位置和所有障碍物
        self.target = self.set_target_position(robot_position)


    def check_collision(self, laser_scan):
        if min(laser_scan) < self.collision_delta:
            return True
        return False

    def check_target(self, distance, collision):
        if distance < self.target_reached_delta and not collision:
            self.target_dist += self.target_dist_increase
            if self.target_dist > self.max_target_dist:
                self.target_dist = self.max_target_dist
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        # Calculate robot heading from odometry data
        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        return distance, cos, sin, angle

    @staticmethod
    def get_reward(goal, collision, action, laser_scan, last_distance, distance):
        if goal:
            return 100.0
        elif collision:
            # return -100.0
            return -160.0
        else:
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
            # return last_distance - distance

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
