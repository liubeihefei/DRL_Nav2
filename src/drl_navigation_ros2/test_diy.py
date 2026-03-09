import torch
import numpy as np
from SAC.SAC import SAC
from ros_python import ROS_env
from utils import record_eval_positions
import time

# Configuration parameters
state_dim = 25
action_dim = 2
max_action = 1
max_steps = 300
eval_cnt = 100
history_n = 1

use_diy_world = True  # 是否使用自定义环境
diy_world_path = "/home/root/rl/DRL_Nav2/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/diy/100by100.model"  # 自定义环境文件路径
obj_cache_path = "/home/root/rl/DRL_Nav2/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/diy/objects.json"  # 物体信息缓存路径
world_size = 100.0   # 自定义环境大小，默认正方形

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SAC(state_dim=state_dim, action_dim=action_dim, max_action=max_action, 
           device=device, save_every=0, load_model=False, history_n=history_n)

# Load weights
model.actor.load_state_dict(torch.load('/home/root/rl/DRL_Nav2/src/drl_navigation_ros2/models/SAC/SAC_actor.pth', map_location=device))
model.actor.eval()
print("Model loaded successfully")

# Initialize environment
ros = ROS_env(
    use_diy_world=use_diy_world,
    diy_world_path=diy_world_path,
    obj_cache_path=obj_cache_path,
    world_size=world_size
)

# Initialize statistics variables
success_count = 0
collision_count = 0
timeout_count = 0
total_steps = 0
total_time = 0

print(f"Starting test for {eval_cnt} times")

# Test loop
cnt = 1
while cnt <= eval_cnt:
    start_time = time.time()
    
    # 重置环境
    latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.reset()
    
    scenario_steps = 0
    scenario_completed = False

    # 多帧时用第一帧进行扩展
    state, _ = model.prepare_state(
        latest_scan, distance, cos, sin, collision, goal, a, vel
    )
    history_state = np.concatenate([state] * history_n)
    
    for count in range(max_steps):
        scenario_steps += 1
        total_steps += 1
        
        # Prepare state
        state, terminal = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a, vel)

        history_state = np.concatenate([history_state[state_dim:], state])

        if terminal:
            break
        
        # # Get action
        # action = model.get_action(state, False)  # False means test mode
        # 多帧历史处理
        action = model.get_action(history_state, False)

        a_in = [(action[0] + 1) / 2, action[1]]
        # # 动作截断
        # a_in = [
        #     # 线速度为[-2.5, 2.5]，截断到[0, 2.5]
        #     (action[0] + 2.5) / 2.0,
        #     # 角速度为[-2.5, 2.5]，缩放到[-1, 1]
        #     action[1] / 2.5,
        # ]
        
        # Execute one step
        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )
        
        # Check if completed
        if goal:
            success_count += 1
            scenario_completed = True
            print(f"Goal reached")
            break
        elif collision:
            collision_count += 1
            scenario_completed = True
            print(f"Collision occurred")
            break
    
    # Check if timeout
    if not scenario_completed and scenario_steps >= max_steps:
        timeout_count += 1
        print(f"Timeout occurred")
    
    # Calculate scenario time
    scenario_time = time.time() - start_time
    total_time += scenario_time
    
    # Progress display
    if cnt % 10 == 0:
        print(f"Completed {cnt}/{eval_cnt} times")

    cnt = cnt + 1

# Calculate statistics
print("\n" + "="*50)
print("Test Results Statistics")
print("="*50)

# Calculate rates
success_rate = success_count / eval_cnt * 100
collision_rate = collision_count / eval_cnt * 100
timeout_rate = timeout_count / eval_cnt * 100

# Print detailed results
print(f"Total test scenarios: {eval_cnt}")
print(f"Success rate: {success_rate:.2f}%")
print(f"Collision rate: {collision_rate:.2f}%")
print(f"Timeout rate: {timeout_rate:.2f}%")
print(f"Total test time: {total_time:.2f} seconds")
