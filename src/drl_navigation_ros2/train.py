from pathlib import Path

from TD3.TD3 import TD3
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining


def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 20  # how many episodes to use to run evaluation
    max_epochs = 100  # max number of epochs
    epoch = 0  # starting epoch number
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 2  # train and update network parameters every n episodes
    train_every_step = 2 # 每多少步更新一次参数
    training_iterations = 2  # how many batches to use for single training cycle
    batch_size = 40  # batch size for each training iteration
    max_steps = 300  # maximum number of steps in single episode
    steps = 0  # starting step number
    load_saved_buffer = False  # whether to load experiences from assets/data.yml
    pretrain = False  # whether to use the loaded experiences to pre-train the model (load_saved_buffer must be True)
    pretraining_iterations = (
        50  # number of training iterations to run during pre-training
    )
    save_every = 100  # save the model every n training cycles
    history_n = 1  # 使用多少帧历史状态，包含当前
    best_success = 0.0  # 记录最好的测试成功率
    best_reward = 0.0  # 记录最好的测试奖励

    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=False,
        history_n=history_n
    )  # instantiate a model

    ros = ROS_env()  # instantiate ROS environment
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    if load_saved_buffer:
        pretraining = Pretraining(
            file_names=["src/drl_navigation_ros2/assets/data.yml"],
            model=model,
            replay_buffer=ReplayBuffer(buffer_size=5e3, random_seed=42),
            reward_function=ros.get_reward,
        )  # instantiate pre-trainind
        replay_buffer = (
            pretraining.load_buffer()
        )  # fill buffer with experiences from the data.yml file
        if pretrain:
            pretraining.train(
                pretraining_iterations=pretraining_iterations,
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # run pre-training
    else:
        replay_buffer = ReplayBuffer(
            buffer_size=5e3, random_seed=42
        )  # if not experiences are loaded, instantiate an empty buffer

    latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state

    # 每个episode是否刚开始收集的标志位，用于重置历史帧
    epi_end_flag = True

    while epoch < max_epochs:  # train until max_epochs is reached
        last_distance = distance

        # 对当前环境进行状态表示
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, vel
        )  # get state a state representation from returned data from the environment

        # 多帧时用第一帧进行扩展
        if epi_end_flag:
            history_state = np.concatenate([state] * history_n)
            # 重置为false，表示此轮不需要再扩展
            epi_end_flag = False

        # 历史进一帧，出一帧
        history_state = np.concatenate([history_state[state_dim:], state])

        # print(history_state)
        # print("--------------------------------\n")

        # 获取动作
        action = model.get_action(history_state, True)  # get an action from the model

        # 动作截断
        a_in = [
            (action[0] + 1) / 2,
            action[1],
        ]  # clip linear velocity to [0, 0.5] m/s range
        
        # 获取下一状态表示
        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1], last_distance=last_distance
        )  # get data from the environment
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, vel
        )  # get a next state representation

        # 未来用历史进一帧，出一帧
        future_state = np.concatenate([history_state[state_dim:], next_state])
        
        replay_buffer.add(
            history_state, action, reward, terminal, future_state
        )  # add experience to the replay buffer

        # 源码单帧状态
        # replay_buffer.add(
        #     state, action, reward, terminal, next_state
        # )  # add experience to the replay buffer
        
        if steps % train_every_step == 0:
            model.train(
                replay_buffer=replay_buffer,
                iterations=training_iterations,
                batch_size=batch_size,
            )  # train the model and update its parameters

        if (
            terminal or steps == max_steps
        ):  # reset environment of terminal stat ereached, or max_steps were taken
            latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.reset()
            episode += 1
            # 收集结束，下个epoch需要进行扩展
            epi_end_flag = True
            # if episode % train_every_n == 0:
            #     model.train(
            #         replay_buffer=replay_buffer,
            #         iterations=training_iterations,
            #          batch_size=batch_size,
            #      )  # train the model and update its parameters

            steps = 0
        else:
            steps += 1

        if (
            episode + 1
        ) % episodes_per_epoch == 0:  # if epoch is concluded, run evaluation
            episode = 0
            epoch += 1
            best_success, best_reward = eval(
                model=model,
                env=ros,
                scenarios=eval_scenarios,
                epoch=epoch,
                max_steps=max_steps,
                state_dim=state_dim,
                history_n=history_n,
                best_success=best_success,
                best_reward=best_reward
            )  # run evaluation


def eval(model, env, scenarios, epoch, max_steps, state_dim, history_n, best_success, best_reward):
    """Function to run evaluation"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
    avg_reward = 0.0
    col = 0
    gl = 0
    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = env.eval(
            scenario=scenario
        )

        # 多帧时用第一帧进行扩展
        state, _ = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, vel
        )
        history_state = np.concatenate([state] * history_n)

        while count < max_steps:
            last_distance = distance

            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a, vel
            )
            
            history_state = np.concatenate([history_state[state_dim:], state])

            if terminal:
                break

            # 源代码单帧处理
            # action = model.get_action(state, False)
            # 多帧历史处理
            action = model.get_action(history_state, False)

            a_in = [(action[0] + 1) / 2, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward, vel = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1], last_distance=last_distance
            )
            avg_reward += reward
            count += 1
            col += collision
            gl += goal
    avg_reward /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)

    # 测试的单独保存
    save_directory=Path("src/drl_navigation_ros2/models/BEST")
    model_name="SAC"
    # 若成功率大于记录，直接保存
    if avg_goal > best_success:
        model.save(model_name, save_directory)
        best_reward = avg_reward
        best_success = avg_goal
    # 若成功率相等，则奖励大于记录时保存
    elif avg_goal == best_success:
        if avg_reward > best_reward:
            model.save(model_name, save_directory)
            best_reward = avg_reward
            best_success = avg_goal
    # 返回值
    print(f"Best Success: {best_success}")
    print(f"Best Reward: {best_reward}")
    return best_success, best_reward


if __name__ == "__main__":
    main()
