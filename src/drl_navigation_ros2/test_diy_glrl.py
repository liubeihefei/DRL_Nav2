import torch
import numpy as np
import time
import math

from SAC.SAC import SAC
from ros_python import ROS_env
import rclpy
from visualization_msgs.msg import Marker

# ===============================
# ⚙️ 参数
# ===============================
state_dim = 25
action_dim = 2
max_action = 1
max_steps = 300
eval_cnt = 20
history_n = 1

astar_resolution = 0.2
robot_radius = 0.4   # ⭐ 建议比真实稍大

waypoint_dist = 2.0
init_target_distance = 8.0

model_path = '/home/horsefly/下载/323/SAC_actor.pth'
diy_world_path = "/home/horsefly/DRL_Nav2/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/diy/40by40.model"
obj_cache_path = "/home/horsefly/DRL_Nav2/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/diy/40by40.json"

# ===============================
# 🧠 A*（基于polygon occupancy）
# ===============================
class AStarPlanner:
    def __init__(self, objects, resolution, rr, world_bounds):
        self.resolution = resolution
        self.rr = rr
        self.objects = objects

        self.min_x = world_bounds[0]
        self.max_x = world_bounds[1]
        self.min_y = world_bounds[0]
        self.max_y = world_bounds[1]

        self.x_width = int((self.max_x - self.min_x) / resolution)
        self.y_width = int((self.max_y - self.min_y) / resolution)

        print("Building occupancy map...")
        self.obstacle_map = self.calc_obstacle_map()
        print("Map build done")

    class Node:
        def __init__(self, x, y, cost, parent):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent = parent

    def planning(self, sx, sy, gx, gy):
        start = self.Node(self.calc_xy(sx, self.min_x),
                          self.calc_xy(sy, self.min_y), 0.0, -1)
        goal = self.Node(self.calc_xy(gx, self.min_x),
                         self.calc_xy(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start)] = start

        while True:
            if not open_set:
                return None

            c_id = min(open_set,
                       key=lambda o: open_set[o].cost +
                       self.calc_heuristic(goal, open_set[o]))
            current = open_set[c_id]

            if current.x == goal.x and current.y == goal.y:
                goal.parent = current.parent
                goal.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for dx, dy, cost in self.motion():
                node = self.Node(current.x + dx,
                                 current.y + dy,
                                 current.cost + cost,
                                 c_id)

                n_id = self.calc_index(node)

                if not self.verify(node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node

        return self.calc_final_path(goal, closed_set)

    def calc_final_path(self, goal, closed):
        rx, ry = [], []
        node = goal
        while node.parent != -1:
            rx.append(self.calc_position(node.x, self.min_x))
            ry.append(self.calc_position(node.y, self.min_y))
            node = closed[node.parent]

        rx.reverse()
        ry.reverse()
        return rx, ry

    def calc_obstacle_map(self):
        grid = [[False]*self.y_width for _ in range(self.x_width)]

        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)

                if self.is_in_obstacle((x, y)):
                    grid[ix][iy] = True

        return grid

    # ⭐ 核心：polygon + 膨胀
    def is_in_obstacle(self, point):
        for obj in self.objects:
            poly = obj.corners_2d

            if self.point_in_polygon(point, poly):
                return True

            if self.point_to_polygon_distance(point, poly) < self.rr:
                return True

        return False

    # ---------- 几何函数 ----------
    def point_to_segment_distance(self, p, a, b):
        ab = (b[0]-a[0], b[1]-a[1])
        ap = (p[0]-a[0], p[1]-a[1])
        ab_len = ab[0]**2 + ab[1]**2
        if ab_len == 0:
            return np.linalg.norm(ap)

        t = (ap[0]*ab[0] + ap[1]*ab[1]) / ab_len

        if t < 0:
            return np.linalg.norm(ap)
        elif t > 1:
            return np.linalg.norm((p[0]-b[0], p[1]-b[1]))
        else:
            proj = (a[0]+t*ab[0], a[1]+t*ab[1])
            return np.linalg.norm((p[0]-proj[0], p[1]-proj[1]))

    def point_to_polygon_distance(self, p, poly):
        return min(self.point_to_segment_distance(p, poly[i], poly[(i+1)%len(poly)])
                   for i in range(len(poly)))

    def point_in_polygon(self, point, poly):
        x, y = point
        inside = False
        j = len(poly)-1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj-xi)*(y-yi)/(yj-yi+1e-6)+xi):
                inside = not inside
            j = i
        return inside

    # ---------- utils ----------
    def verify(self, node):
        if node.x < 0 or node.y < 0:
            return False
        if node.x >= self.x_width or node.y >= self.y_width:
            return False
        return not self.obstacle_map[node.x][node.y]

    def calc_xy(self, pos, minp):
        return int((pos - minp) / self.resolution)

    def calc_position(self, index, minp):
        return index * self.resolution + minp

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    def calc_heuristic(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def save_map(self, prefix="astar_map"):
        import numpy as np
        import matplotlib.pyplot as plt

        grid = np.array(self.obstacle_map, dtype=np.uint8)

        # 1️⃣ 保存原始grid
        np.save(f"{prefix}.npy", grid)

        # 2️⃣ 保存可视化
        plt.imshow(grid.T, origin='lower', cmap='gray')
        plt.title("Occupancy Map")
        plt.savefig(f"{prefix}.png")
        plt.close()

        print(f"Map saved: {prefix}.npy / {prefix}.png")

    @staticmethod
    def motion():
        return [
            (1,0,1),(0,1,1),(-1,0,1),(0,-1,1),
            (1,1,1.4),(1,-1,1.4),(-1,1,1.4),(-1,-1,1.4)
        ]


# ===============================
# 📌 路径稀疏
# ===============================
def sparse_path(rx, ry, interval):
    path = list(zip(rx, ry))
    new_path = [path[0]]
    acc = 0.0

    for i in range(1, len(path)):
        d = math.hypot(path[i][0]-path[i-1][0],
                       path[i][1]-path[i-1][1])
        acc += d
        if acc >= interval:
            new_path.append(path[i])
            acc = 0

    if new_path[-1] != path[-1]:
        new_path.append(path[-1])

    return new_path


# ===============================
# 📍 waypoint 可视化
# ===============================
class WaypointPublisher:
    def __init__(self):
        self.node = rclpy.create_node('wp_pub')
        self.pub = self.node.create_publisher(Marker, '/waypoints', 10)

    def publish(self, wps):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.node.get_clock().now().to_msg()

        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST   # ⭐ 推荐
        marker.action = Marker.ADD

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        from geometry_msgs.msg import Point
        for p in wps:
            pt = Point()
            pt.x, pt.y = p
            pt.z = 0.0
            marker.points.append(pt)

        self.pub.publish(marker)


# ===============================
# 🚀 主程序
# ===============================
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SAC(state_dim=state_dim, action_dim=action_dim, max_action=max_action, 
           device=device, save_every=0, load_model=False, history_n=history_n)

# Load weights
model.actor.load_state_dict(torch.load(model_path, map_location=device))
model.actor.eval()
print("Model loaded successfully")

ros = ROS_env(
    init_target_distance=init_target_distance,
    use_diy_world=True,
    diy_world_path=diy_world_path,
    obj_cache_path=obj_cache_path,
    world_size=38.0,
    min_pose_distance=0.5
)

# ⭐ 只初始化一次地图
planner = AStarPlanner(
    ros.objects,
    astar_resolution,
    robot_radius,
    ros.world_bounds
)
planner.save_map("map_40x40")

wp_pub = WaypointPublisher()

# ===============================
# 🎯 测试
# ===============================
for ep in range(eval_cnt):

    while True:
        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.reset()

        # 保证拿到最新的机器人位置
        rclpy.spin_once(ros.sensor_subscriber)

        sx, sy = ros.sensor_subscriber.get_latest_sensor()[1].x, \
                ros.sensor_subscriber.get_latest_sensor()[1].y
        gx, gy = ros.target

        # ❗ 可选：过滤非法点（强烈建议）
        if planner.is_in_obstacle((sx, sy)) or planner.is_in_obstacle((gx, gy)):
            # print("Start/Goal in obstacle → resample")
            continue

        result = planner.planning(sx, sy, gx, gy)

        if result is None:
            # print("A* failed → resample")
            continue

        rx, ry = result
        break

    if rx is None:
        print("A* failed")
        continue

    waypoints = sparse_path(rx, ry, waypoint_dist)
    wp_pub.publish(waypoints)

    wp_idx = 0

    state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a, vel)
    history_state = np.concatenate([state]*history_n)

    for step in range(int(max_steps * len(waypoints) * waypoint_dist / 8.0)):

        wx, wy = waypoints[wp_idx]
        ros.target = [wx, wy]

        state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a, vel)
        history_state = np.concatenate([history_state[state_dim:], state])

        action = model.get_action(history_state, False)
        a_in = [(action[0]+1)/2, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        if goal:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                print("Goal reached")
                break

        if collision:
            print("Collision")
            break

    print(f"Episode {ep} done")