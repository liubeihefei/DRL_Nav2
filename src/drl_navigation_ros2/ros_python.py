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

# 添加环境解析工具
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from world_parse import ObjectInfo, WorldParser


# 仿真环境类
class ROS_env:
    def __init__(
        self,
        init_target_distance=2.0,   # 初始目标距离
        target_dist_increase=0.001, # 每次达到目标后目标距离增加的值
        max_target_dist=8.0,        # 目标距离的最大值
        target_reached_delta=0.5,   # 到达目标的距离阈值
        collision_delta=0.4,        # 发生碰撞的距离阈值
        use_diy_world=False,        # 是否使用自定义环境
        diy_world_path=None,        # 自定义环境文件路径
        obj_cache_path=None,        # 物体信息缓存路径
        world_size=100.0,           # 自定义环境大小，默认正方形，单位为米
        min_pose_distance=1.8,      # 生成机器人和目标位置时与障碍物的最小距离
        args=None,
    ):
        rclpy.init(args=args)
        # 是否使用自定义环境、自定义环境文件路径、自定义环境边界
        self.use_diy_world = use_diy_world
        self.diy_world_path = diy_world_path
        self.obj_cache_path = obj_cache_path
        self.world_bounds = [-world_size / 2.0, world_size / 2.0]
        self.min_pose_distance = min_pose_distance

        # 如果使用自定义环境，解析环境文件获取初始物体位置和外接框
        if self.use_diy_world:
            if self.obj_cache_path is None:
                self.parser = WorldParser(os.path.expanduser("~/.gazebo/models"))
                self.objects = self.parser.parse_world_file(self.diy_world_path)
            else:
                self.parser = WorldParser(os.path.expanduser("~/.gazebo/models"))
                self.objects = self.parser.load_objects(self.obj_cache_path)

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
    def step(self, steps=-1, max_steps=300, lin_velocity=0.0, ang_velocity=0.1, last_distance=0.0):
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
        reward = self.get_reward(steps, max_steps, goal, collision, action, latest_scan, last_distance, distance)

        # 返回所有状态所需的数据、是否碰撞、是否到达、奖励
        return latest_scan, distance, cos, sin, collision, goal, action, reward, latest_vel

    # 重置环境
    def reset(self):
        if not self.use_diy_world:
            # 重置世界，此时机器人仅是位置复原，可能还有速度
            self.world_reset.reset_world()
        # 发布0速度让机器人停下
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )

        # 若使用自定义环境
        if self.use_diy_world:
            # 生成机器人和目标位置
            robot_pos, target_pos = self.generate_robot_and_target()
            
            # 设置机器人位置
            robot_angle = np.random.uniform(-np.pi, np.pi)
            self.set_position("turtlebot3_waffle", robot_pos[0], robot_pos[1], robot_angle)
            # 设置目标
            self.target = list(target_pos)

        # 若使用源码环境
        else:
            # 重置障碍物位置记录（如源码设置一次环境后是8个随机障碍物，但此时需要变回4个），为后面重新设置随机障碍物等位置做准备
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


    # 计算点到线段的最小距离
    def point_to_segment_distance(self, p, a, b):
        # 向量AB
        ab = (b[0] - a[0], b[1] - a[1])
        # 向量AP
        ap = (p[0] - a[0], p[1] - a[1])
        
        # 计算投影参数t
        ab_len_sq = ab[0]**2 + ab[1]**2
        if ab_len_sq == 0:
            return np.linalg.norm(ap)
        
        t = (ap[0]*ab[0] + ap[1]*ab[1]) / ab_len_sq
        
        if t < 0:
            # 投影在A点之外
            return np.linalg.norm(ap)
        elif t > 1:
            # 投影在B点之外
            bp = (p[0] - b[0], p[1] - b[1])
            return np.linalg.norm(bp)
        else:
            # 投影在线段上
            projection = (a[0] + t*ab[0], a[1] + t*ab[1])
            return np.linalg.norm((p[0] - projection[0], p[1] - projection[1]))

    # 计算点到多边形的最小距离
    def point_to_polygon_distance(self, point, polygon):
        # 计算点到多边形每条边的距离
        min_dist = float('inf')
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            dist = self.point_to_segment_distance(point, a, b)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    # 判断点是否在多边形内部（射线法）
    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            
            # 检查射线是否与边相交
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    # 检查点是否与任何障碍物发生碰撞
    def check_point_collision(self, point):
        for obj in self.objects:
            # 判断点是否在物体内部
            if self.point_in_polygon(point, obj.corners_2d):
                return False

            # 判断点是否离物体太近
            dist = self.point_to_polygon_distance(point, obj.corners_2d)
            if dist < self.min_pose_distance:
                return False  
        
        return True  # 点有效
    
    # 随机生成一个位置，确保与所有障碍物的距离大于阈值
    def generate_random_position(self):
        while True:
            # 随机生成一个位置
            x = np.random.uniform(self.world_bounds[0], self.world_bounds[1])
            y = np.random.uniform(self.world_bounds[0], self.world_bounds[1])
            
            # 判断是否发生碰撞，包括在障碍物内部和距离障碍物太近两种情况
            if self.check_point_collision((x, y)):
                return (x, y)
    
    # 生成机器人位置，确保与所有障碍物的距离大于距离阈值
    def generate_robot_position(self):
        return self.generate_random_position()
    
    # 生成目标位置，确保与机器人距离为目标距离，并且与所有障碍物的距离大于距离阈值
    def generate_target_position(self, robot_position):
        while True:
            # 计算目标位置
            x = np.clip(
                robot_position[0]
                + np.random.uniform(-self.target_dist, self.target_dist),
                self.world_bounds[0], self.world_bounds[1]
            )
            y = np.clip(
                robot_position[1]
                + np.random.uniform(-self.target_dist, self.target_dist),
                self.world_bounds[0], self.world_bounds[1]
            )
            
            # 检查目标位置是否有效
            if self.check_point_collision((x, y)):
                return (x, y)
    
    # 生成机器人和目标位置
    def generate_robot_and_target(self):
        # 先生成机器人位置
        robot_pos = self.generate_robot_position()
        
        # 然后生成目标位置
        target_pos = self.generate_target_position(robot_pos)
        
        return robot_pos, target_pos


    # @staticmethod
    # def get_reward(goal, collision, action, laser_scan, last_distance, distance):
    #     if goal:
    #         return 100.0
    #     elif collision:
    #         # return -100.0
    #         return -160.0
    #     else:
    #         r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
    #         return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
    #         # return last_distance - distance
        
    @staticmethod
    def get_reward(steps, max_steps, goal, collision, action, laser_scan, last_distance, distance):
        if goal:
            return 10.0
        elif collision:
            return -10.0
        else:
            if (steps > max_steps - max_steps / 30.0) or (random.random() < 0.1):
                r_task = 1.0 / ( 1.0 + abs(distance / 4.0))
                return r_task
            else:
                return 0.0

    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
