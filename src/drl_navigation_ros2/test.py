import torch
import numpy as np
from SAC.SAC import SAC
from ros_python import ROS_env
from utils import record_eval_positions
import time

# Configuration parameters
state_dim = 25
action_dim = 2
max_action = 1.0
max_steps = 300
scenarios_nums = 1000
history_n = 1

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SAC(state_dim=state_dim, action_dim=action_dim, max_action=max_action, 
           device=device, save_every=0, load_model=False, history_n=history_n)

# Load weights
model.actor.load_state_dict(torch.load('/home/horsefly/下载/BEST/SAC_actor.pth', map_location=device))
model.actor.eval()
print("Model loaded successfully")

# Initialize environment and generate scenarios
ros = ROS_env()
eval_scenarios = record_eval_positions(n_eval_scenarios=scenarios_nums)

# Initialize statistics variables
success_count = 0
collision_count = 0
timeout_count = 0
total_steps = 0
total_time = 0

print(f"Starting test for {len(eval_scenarios)} scenarios")

# Test loop
for i, scenario in enumerate(eval_scenarios):
    start_time = time.time()
    
    # Reset scenario
    latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.eval(scenario=scenario)
    
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
    if (i + 1) % 10 == 0:
        print(f"Completed {i+1}/{len(eval_scenarios)} scenarios")

# Calculate statistics
print("\n" + "="*50)
print("Test Results Statistics")
print("="*50)

# Calculate rates
success_rate = success_count / scenarios_nums * 100
collision_rate = collision_count / scenarios_nums * 100
timeout_rate = timeout_count / scenarios_nums * 100

# Print detailed results
print(f"Total test scenarios: {scenarios_nums}")
print(f"Success rate: {success_rate:.2f}%")
print(f"Collision rate: {collision_rate:.2f}%")
print(f"Timeout rate: {timeout_rate:.2f}%")
print(f"Total test time: {total_time:.2f} seconds")
