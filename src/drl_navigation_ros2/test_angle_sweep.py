"""
固定距离 + 可控初始夹角的导航测试脚本
=====================================

功能:
    1. 使用源码自带的 10x10 环境，但把所有障碍物挪到远处，等价于"无障碍"。
    2. 推理相关参数(model_path / state_dim / history_n 等)统一在文件顶部 CONFIG 区配置。
    3. INITIAL_ANGLE_DEG: 机器人初始朝向与"机器人->目标连线"的夹角(单位: 度)。
    4. TARGET_DISTANCE  : 机器人到目标点的固定直线距离(米)。每轮均严格固定。
    5. 每次到达目标时累计实际行驶过的路径长度并打印。
    6. SWEEP_MODE + ANGLE_LIST + RUNS_PER_ANGLE:
       打开 SWEEP_MODE 后, 按 ANGLE_LIST 逐个角度跑 RUNS_PER_ANGLE 次,
       记录平均路径长度并绘制 "平均路径长度 vs 初始夹角" 关系图。

约定:
    * 机器人放在原点 (0, 0)。
    * 目标点放在 (TARGET_DISTANCE, 0), 即位于 +x 方向。
    * 机器人航向角 = θ(弧度) ⇒ 机器人面向 与 目标方向(+x) 的夹角即 θ。
"""

import json
import math
import os
import time

import numpy as np
import torch
import rclpy

import matplotlib
# 服务器/无显示环境下也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SAC.SAC import SAC
from ros_python import ROS_env


# ============================================================
# ⚙️ 全局配置 (所有要改的地方都在这里)
# ============================================================

# ---------- 推理 / 模型相关 (与其它 test 脚本保持一致) ----------
MODEL_PATH   = "/home/horsefly/下载/BEST/SAC_actor.pth"  # SAC actor 权重路径(.pth)
STATE_DIM    = 25
ACTION_DIM   = 2
MAX_ACTION   = 1.0
HISTORY_N    = 1     # 历史帧数, 必须与训练时一致
MAX_STEPS    = 300   # 单回合最大 step 数

# ---------- 环境 / 任务相关 ----------
TARGET_DISTANCE      = 4.0   # 机器人与目标点的固定直线距离(米)
TARGET_REACHED_DELTA = 0.5   # 到达目标的距离阈值(米)
COLLISION_DELTA      = 0.4   # 雷达判定碰撞的距离阈值(米)
OBSTACLE_PARK_POS    = 80.0  # 把所有障碍物挪到 (~N, ~N) 远处, N 越大越彻底

# ---------- 单次 / 多次测试相关 ----------
# 机器人初始朝向 与 机器人->目标方向 的夹角(度).
# 正值: 逆时针; 负值: 顺时针. 仅在 SWEEP_MODE = False 时使用.
INITIAL_ANGLE_DEG    = 0.0
RUNS_PER_ANGLE       = 1     # 每个夹角下重复测试的次数

# ---------- 角度扫描模式 ----------
SWEEP_MODE   = False                          # True 时进行多角度扫描并出图
ANGLE_LIST   = [0, 30, 60, 90, 120, 150, 180] # 扫描的夹角列表(度)

# ---------- 输出 ----------
PLOT_PATH    = "angle_vs_path.png"   # 角度-平均路径长度关系图保存路径
CSV_PATH     = "angle_vs_path.csv"   # 每次测试一行的原始结果 CSV
JSON_PATH    = ""                    # 汇总结果 JSON; 空字符串表示不保存

# ============================================================


# ============================================================
# 工具函数: 把所有障碍物挪到远处
# ============================================================
def park_all_obstacles(ros, far=OBSTACLE_PARK_POS):
    """
    源码 10x10 世界中存在 obstacle1 ~ obstacle8 (4 固定 + 4 随机)。
    把它们全部挪到 (>>10) 远处, 等价于"无障碍"。
    任何不存在的 entity 调用 set_state 不会抛异常(call_async)，安全可忽略。
    """
    for i in range(1, 9):
        x = far + i * 3.0
        y = far + i * 3.0
        try:
            ros.set_position(f"obstacle{i}", x, y, 0.0)
        except Exception:
            # 静默忽略
            pass
    # 给一点点时间让 set_entity_state 生效
    time.sleep(0.05)


# ============================================================
# 自定义 reset: 固定起点/终点/初始夹角
# ============================================================
def custom_reset(ros, target_distance, initial_angle_deg):
    """
    手动复位环境:
      * 机器人放在 (0, 0), 朝向角 = initial_angle_deg (度)
      * 目标固定放在 (target_distance, 0)
    返回与 ros.reset() 兼容的元组。
    """
    # 1) 让机器人停下
    ros.cmd_vel_publisher.publish_cmd_vel(0.0, 0.0)
    ros.momentum_lin = 0.0
    ros.momentum_ang = 0.0

    # 2) 把所有障碍物移到远处 (每回合都执行, 防止万一被复位)
    park_all_obstacles(ros, far=OBSTACLE_PARK_POS)

    # 3) 设置目标点
    target_x = float(target_distance)
    target_y = 0.0
    ros.target = [target_x, target_y]
    ros.publish_target.publish(target_x, target_y)

    # 4) 设置机器人位姿
    initial_angle_rad = math.radians(initial_angle_deg)
    ros.set_position("turtlebot3_waffle", 0.0, 0.0, initial_angle_rad)

    # 5) 等位置生效, 然后发空 step 拿到初始状态
    time.sleep(0.2)
    ros.physics_client.unpause_physics()
    time.sleep(0.2)
    rclpy.spin_once(ros.sensor_subscriber)
    ros.physics_client.pause_physics()

    latest_scan, distance, cos, sin, _, _, action, reward, vel = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )
    return latest_scan, distance, cos, sin, False, False, action, reward, vel


# ============================================================
# 单次 episode
# ============================================================
def run_one_episode(ros, model, initial_angle_deg, ep_tag=""):
    """
    返回 dict: { angle_deg, success, collision, timeout, path_length, steps, time }
    """
    t0 = time.time()
    latest_scan, distance, cos, sin, collision, goal, a, reward, vel = custom_reset(
        ros, TARGET_DISTANCE, initial_angle_deg
    )

    # 初始化路径长度统计
    pos = ros.sensor_subscriber.latest_position
    if pos is None:
        last_xy = (0.0, 0.0)
    else:
        last_xy = (pos.x, pos.y)
    path_length = 0.0

    # history_state 初始化(模仿其它 test 脚本)
    state, _ = model.prepare_state(latest_scan, distance, cos, sin,
                                   collision, goal, a, vel)
    history_state = np.concatenate([state] * HISTORY_N)

    success = False
    collided = False
    steps_taken = 0

    for step in range(MAX_STEPS):
        steps_taken = step + 1

        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a, vel
        )
        history_state = np.concatenate(
            [history_state[STATE_DIM:], state]
        )

        if terminal:
            break

        action = model.get_action(history_state, False)
        a_in = [(action[0] + 1) / 2, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        # 增量累积路径长度
        pos = ros.sensor_subscriber.latest_position
        if pos is not None:
            cur_xy = (pos.x, pos.y)
            d_seg = math.hypot(cur_xy[0] - last_xy[0],
                               cur_xy[1] - last_xy[1])
            path_length += d_seg
            last_xy = cur_xy

        if goal:
            success = True
            break
        if collision:
            collided = True
            break

    timeout = (not success) and (not collided)
    elapsed = time.time() - t0

    print(f"  [{ep_tag}] angle={initial_angle_deg:>+7.2f}°  "
          f"success={int(success)}  collision={int(collided)}  "
          f"timeout={int(timeout)}  "
          f"path_len={path_length:.3f} m  steps={steps_taken}  "
          f"time={elapsed:.1f}s")

    return {
        "angle_deg": float(initial_angle_deg),
        "success": bool(success),
        "collision": bool(collided),
        "timeout": bool(timeout),
        "path_length": float(path_length),
        "steps": int(steps_taken),
        "time_sec": float(elapsed),
    }


# ============================================================
# 绘图
# ============================================================
def plot_angle_vs_path(angle_list, avg_path_lengths, success_rates,
                       target_distance, runs_per_angle, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # 上图: 平均路径长度
    ax = axes[0]
    ax.plot(angle_list, avg_path_lengths, marker="o", color="steelblue",
            label="Avg path length (success only)")
    ax.axhline(target_distance, ls="--", color="gray",
               label=f"Straight-line distance ({target_distance} m)")
    ax.set_ylabel("Average path length (m)")
    ax.set_title(
        f"Avg path length vs initial heading angle\n"
        f"target_distance = {target_distance} m, "
        f"runs per angle = {runs_per_angle}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 下图: 成功率
    ax = axes[1]
    ax.bar(angle_list, success_rates,
           width=max(2.0, (max(angle_list) - min(angle_list))
                     / max(1, len(angle_list)) * 0.6),
           color="seagreen", alpha=0.8)
    ax.set_xlabel("Initial heading angle (deg)")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[Plot] saved to {out_path}")


# ============================================================
# 结果保存
# ============================================================
def save_csv(rows, csv_path):
    if not csv_path:
        return
    header = ["angle_deg", "run_idx", "success", "collision", "timeout",
              "path_length", "steps", "time_sec"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")
    print(f"[CSV ] saved to {csv_path}")


def save_json(summary, json_path):
    if not json_path:
        return
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[JSON] saved to {json_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    # 决定要测试的角度列表
    if SWEEP_MODE:
        angle_list = list(ANGLE_LIST)
    else:
        angle_list = [INITIAL_ANGLE_DEG]

    # ---------- 加载模型 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SAC(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        max_action=MAX_ACTION, device=device,
        save_every=0, load_model=False, history_n=HISTORY_N,
    )
    model.actor.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.actor.eval()
    print(f"Model loaded from: {MODEL_PATH}")

    # ---------- 初始化环境 ----------
    # 用源码 10x10 世界, 关闭目标距离自增, 把目标距离写死成 TARGET_DISTANCE
    ros = ROS_env(
        init_target_distance=TARGET_DISTANCE,
        target_dist_increase=0.0,
        max_target_dist=TARGET_DISTANCE,
        target_reached_delta=TARGET_REACHED_DELTA,
        collision_delta=COLLISION_DELTA,
        use_diy_world=False,
    )

    # 启动一次, 把障碍挪走
    ros.physics_client.unpause_physics()
    time.sleep(0.3)
    park_all_obstacles(ros, far=OBSTACLE_PARK_POS)
    time.sleep(0.3)
    ros.physics_client.pause_physics()

    # ---------- 跑测试 ----------
    all_rows = []                                # 每次测试一行
    summary_per_angle = []                       # 每个角度一项
    avg_path_per_angle = []
    success_rate_per_angle = []

    print(f"\nStart sweeping {len(angle_list)} angle(s), "
          f"{RUNS_PER_ANGLE} run(s) each, "
          f"target_distance={TARGET_DISTANCE} m\n")

    for angle in angle_list:
        print(f"=== Angle {angle:.2f}° ===")
        per_angle_results = []
        for run_idx in range(RUNS_PER_ANGLE):
            tag = f"{angle:>+7.2f}° / run {run_idx+1}/{RUNS_PER_ANGLE}"
            res = run_one_episode(ros, model, angle, ep_tag=tag)
            res["run_idx"] = run_idx
            all_rows.append(res)
            per_angle_results.append(res)

        # ---- 该角度的统计 ----
        n_total = len(per_angle_results)
        success_results = [r for r in per_angle_results if r["success"]]
        n_success = len(success_results)
        success_rate = 100.0 * n_success / max(1, n_total)

        if n_success > 0:
            avg_path = float(np.mean([r["path_length"] for r in success_results]))
            std_path = float(np.std([r["path_length"] for r in success_results]))
        else:
            avg_path = float("nan")
            std_path = float("nan")

        # 也保留所有(含失败)的平均, 便于交叉对比
        avg_path_all = float(np.mean([r["path_length"]
                                      for r in per_angle_results]))

        avg_path_per_angle.append(avg_path)
        success_rate_per_angle.append(success_rate)

        summary_per_angle.append({
            "angle_deg": float(angle),
            "n_total": n_total,
            "n_success": n_success,
            "n_collision": sum(1 for r in per_angle_results if r["collision"]),
            "n_timeout":   sum(1 for r in per_angle_results if r["timeout"]),
            "success_rate_pct": success_rate,
            "avg_path_length_success": avg_path,
            "std_path_length_success": std_path,
            "avg_path_length_all": avg_path_all,
        })
        print(f"--- angle {angle:.2f}°: success_rate={success_rate:.1f}%, "
              f"avg_path(success)={avg_path:.3f} m, "
              f"avg_path(all)={avg_path_all:.3f} m\n")

    # ---------- 总结输出 ----------
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'angle(deg)':>12}  {'success%':>10}  "
          f"{'avg_path(succ)':>16}  {'avg_path(all)':>15}")
    for s in summary_per_angle:
        print(f"{s['angle_deg']:>12.2f}  "
              f"{s['success_rate_pct']:>10.1f}  "
              f"{s['avg_path_length_success']:>16.3f}  "
              f"{s['avg_path_length_all']:>15.3f}")

    # ---------- 保存 CSV / JSON ----------
    save_csv(all_rows, CSV_PATH)
    save_json({
        "config": {
            "MODEL_PATH": MODEL_PATH,
            "STATE_DIM": STATE_DIM,
            "ACTION_DIM": ACTION_DIM,
            "MAX_ACTION": MAX_ACTION,
            "HISTORY_N": HISTORY_N,
            "MAX_STEPS": MAX_STEPS,
            "TARGET_DISTANCE": TARGET_DISTANCE,
            "TARGET_REACHED_DELTA": TARGET_REACHED_DELTA,
            "COLLISION_DELTA": COLLISION_DELTA,
            "OBSTACLE_PARK_POS": OBSTACLE_PARK_POS,
            "INITIAL_ANGLE_DEG": INITIAL_ANGLE_DEG,
            "RUNS_PER_ANGLE": RUNS_PER_ANGLE,
            "SWEEP_MODE": SWEEP_MODE,
            "ANGLE_LIST": ANGLE_LIST,
        },
        "angle_list": angle_list,
        "summary_per_angle": summary_per_angle,
    }, JSON_PATH)

    # ---------- 出图 (sweep 模式且至少 2 个角度才有意义) ----------
    if SWEEP_MODE and len(angle_list) >= 2:
        plot_angle_vs_path(
            angle_list=angle_list,
            avg_path_lengths=avg_path_per_angle,
            success_rates=success_rate_per_angle,
            target_distance=TARGET_DISTANCE,
            runs_per_angle=RUNS_PER_ANGLE,
            out_path=PLOT_PATH,
        )

    # ---------- 收尾 ----------
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
