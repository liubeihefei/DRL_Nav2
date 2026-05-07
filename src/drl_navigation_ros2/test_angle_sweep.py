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
PLOT_PATH    = "angle_vs_path.png"   # 综合图(顶部 path length + 下方各角度轨迹)
CSV_PATH     = "angle_vs_path.csv"   # 每次测试一行的原始结果 CSV
JSON_PATH    = ""                    # 汇总结果 JSON; 空字符串表示不保存

# 轨迹图布局: 每行最多放几个子图
TRAJ_MAX_COLS = 4

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
    返回 dict: { angle_deg, success, collision, timeout, path_length,
                 steps, time, trajectory }
    trajectory 为 [(x, y), ...] 列表(机器人逐步的 odom 位置)。

    注意: 路径长度 = Σ ||p_t - p_{t-1}|| (分段折线长度), 是真实曲线长度的下界
    估计(弦 <= 弧). 由于到达判定阈值是 TARGET_REACHED_DELTA(默认 0.5m), 即
    distance < TARGET_REACHED_DELTA 就算成功, 所以即使完全沿直线冲向目标,
    实际也只走了 TARGET_DISTANCE - TARGET_REACHED_DELTA, 这是直线情形下
    "理论最短路径长度", 不是 TARGET_DISTANCE.
    """
    t0 = time.time()
    latest_scan, distance, cos, sin, collision, goal, a, reward, vel = custom_reset(
        ros, TARGET_DISTANCE, initial_angle_deg
    )

    # 初始化路径长度 + 轨迹统计
    # 多 spin 几次, 让 odom 跟上 set_position 之后的瞬移
    for _ in range(3):
        rclpy.spin_once(ros.sensor_subscriber, timeout_sec=0.05)
    pos = ros.sensor_subscriber.latest_position
    if pos is None:
        last_xy = (0.0, 0.0)
    else:
        last_xy = (pos.x, pos.y)
    path_length = 0.0
    trajectory = [last_xy]

    # 每个 step 的指令角速度 (策略实际下发的 a_in[1], 单位 rad/s)
    ang_vel_cmd = []

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

        # 记录本 step 实际下发的角速度
        ang_vel_cmd.append(float(a_in[1]))

        latest_scan, distance, cos, sin, collision, goal, a, reward, vel = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )

        # 增量累积路径长度 + 记录轨迹点
        pos = ros.sensor_subscriber.latest_position
        if pos is not None:
            cur_xy = (pos.x, pos.y)
            d_seg = math.hypot(cur_xy[0] - last_xy[0],
                               cur_xy[1] - last_xy[1])
            path_length += d_seg
            last_xy = cur_xy
            trajectory.append(cur_xy)

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
        "trajectory": trajectory,        # [(x, y), ...]
        "ang_vel_cmd": ang_vel_cmd,      # 每 step 的指令角速度 (rad/s)
    }


# ============================================================
# 绘图
# ============================================================
def plot_combined(angle_list, avg_path_lengths, results_per_angle,
                  target_distance, target_reached_delta,
                  runs_per_angle, out_path,
                  traj_max_cols=TRAJ_MAX_COLS):
    """
    一张图里上下两块:
      * 顶部一行: 平均路径长度 vs 初始夹角
      * 下方网格: 每个角度一个子图, 画出该角度下所有 run 的轨迹
                  (起点 + 终点圆 + 折线连接的轨迹点)
    """
    n_angles = len(angle_list)
    n_cols = max(1, min(traj_max_cols, n_angles))
    n_rows = (n_angles + n_cols - 1) // n_cols

    # 网格布局:
    #   row 0:                 顶部 path length(全宽)
    #   每个 "block" 占两行:   上 = 轨迹 (3.4), 下 = 角速度曲线 (1.8)
    # 共 n_rows 个 block ⇒ 总行数 = 1 + 2*n_rows
    height_ratios = [4.0]
    for _ in range(n_rows):
        height_ratios.append(3.4)   # 轨迹
        height_ratios.append(1.8)   # 角速度
    fig_h = sum(height_ratios) + 1.0
    fig_w = max(8.0, 3.2 * n_cols)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1 + 2 * n_rows, n_cols,
        height_ratios=height_ratios,
        hspace=0.55, wspace=0.3,
    )

    # ---------- 顶部: 平均路径长度 ----------
    ax_path = fig.add_subplot(gs[0, :])
    ax_path.plot(angle_list, avg_path_lengths, marker="o",
                 color="steelblue", linewidth=1.5,
                 label="Avg path length (success only)")
    ax_path.axhline(target_distance, ls="--", color="gray",
                    label=f"Straight-line distance ({target_distance} m)")
    min_eff = target_distance - target_reached_delta
    ax_path.axhline(min_eff, ls=":", color="orange",
                    label=f"Theoretical min path "
                          f"(distance - reach_delta = {min_eff:.2f} m)")
    ax_path.set_xlabel("Initial heading angle (deg)")
    ax_path.set_ylabel("Average path length (m)")
    ax_path.set_title(
        f"Avg path length vs initial heading angle\n"
        f"target_distance = {target_distance} m, "
        f"reach_delta = {target_reached_delta} m, "
        f"runs per angle = {runs_per_angle}"
    )
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(loc="best", fontsize=9)

    # ---------- 下方: 每个角度一个轨迹子图 ----------
    # 计算所有轨迹的全局边界, 让子图统一坐标范围
    all_xs, all_ys = [0.0, target_distance], [0.0]
    for runs in results_per_angle:
        for r in runs:
            for (x, y) in r["trajectory"]:
                all_xs.append(x)
                all_ys.append(y)
    pad = max(0.5, 0.1 * target_distance)
    x_min, x_max = min(all_xs) - pad, max(all_xs) + pad
    y_min, y_max = min(all_ys) - pad, max(all_ys) + pad
    # 让 x/y 同等比例视觉范围, 不强制 equal
    span = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x_min, x_max = cx - span / 2.0, cx + span / 2.0
    y_min, y_max = cy - span / 2.0, cy + span / 2.0

    cmap = plt.get_cmap("tab10")

    # 计算所有 ang_vel 的全局 y 范围, 让所有角速度子图统一比例
    all_av = []
    for runs in results_per_angle:
        for r in runs:
            all_av.extend(r.get("ang_vel_cmd", []))
    if len(all_av) > 0:
        av_max = max(abs(min(all_av)), abs(max(all_av)))
    else:
        av_max = 1.0
    av_max = max(av_max, 0.1) * 1.1   # 留点余量, 避免顶到边

    for idx, angle in enumerate(angle_list):
        block_row = idx // n_cols
        col = idx % n_cols
        traj_row = 1 + 2 * block_row     # 轨迹行
        av_row   = 2 + 2 * block_row     # 角速度行

        # ---------- 轨迹子图 ----------
        ax = fig.add_subplot(gs[traj_row, col])

        runs = results_per_angle[idx]
        for run_idx, r in enumerate(runs):
            traj = r["trajectory"]
            if len(traj) == 0:
                continue
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            color = cmap(run_idx % 10)
            # 细线连接 + 小点
            ax.plot(xs, ys, "-", color=color, linewidth=0.8, alpha=0.85)
            ax.plot(xs, ys, ".", color=color, markersize=2.5, alpha=0.9)
            # 终点用空心圈标一下, 区分成功/失败
            end_marker = "o" if r["success"] else ("x" if r["collision"] else "s")
            ax.plot(xs[-1], ys[-1], end_marker, color=color,
                    markersize=6, markerfacecolor="none", markeredgewidth=1.2)

        # 起点
        ax.plot(0.0, 0.0, marker="P", color="green",
                markersize=10, label="start" if idx == 0 else None)
        # 目标点
        ax.plot(target_distance, 0.0, marker="*", color="red",
                markersize=12, label="goal" if idx == 0 else None)
        # 到达判定圈
        circle = plt.Circle((target_distance, 0.0), target_reached_delta,
                            fill=False, ls="--", color="red", linewidth=0.8,
                            alpha=0.6)
        ax.add_patch(circle)

        # 画一条 start -> goal 的灰色参考线
        ax.plot([0.0, target_distance], [0.0, 0.0],
                ls=":", color="gray", linewidth=0.7, alpha=0.6)

        # 用一个箭头标出初始朝向
        arr_len = max(0.3, 0.15 * target_distance)
        ang_rad = math.radians(angle)
        ax.annotate(
            "", xy=(arr_len * math.cos(ang_rad), arr_len * math.sin(ang_rad)),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.4),
        )

        ax.set_title(f"angle = {angle:g}°  (n={len(runs)})", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("y (m)")
        # 轨迹子图的 x 标签由下方的角速度子图代替, 这里不画 xlabel

        # ---------- 角速度子图 (放在该角度轨迹的正下方) ----------
        ax_av = fig.add_subplot(gs[av_row, col])
        for run_idx, r in enumerate(runs):
            av = r.get("ang_vel_cmd", [])
            if len(av) == 0:
                continue
            color = cmap(run_idx % 10)
            steps = np.arange(1, len(av) + 1)
            ax_av.plot(steps, av, "-", color=color, linewidth=0.9, alpha=0.85)
            ax_av.plot(steps, av, ".", color=color, markersize=1.5, alpha=0.7)

        # 0 参考线 + ±0.3 红色虚线参考(用于观察转弯幅度)
        ax_av.axhline(0.0, color="gray", linewidth=0.6, alpha=0.6)
        ax_av.axhline(0.3, color="red", linewidth=0.8,
                      linestyle="--", alpha=0.7)
        ax_av.axhline(-0.3, color="red", linewidth=0.8,
                      linestyle="--", alpha=0.7)

        ax_av.set_ylim(-av_max, av_max)
        ax_av.grid(True, alpha=0.3)

        # 显式设定刻度: 让 ±0.3 一定有刻度, 同时保留 0 和上下界
        yticks = sorted({-round(av_max, 2), -0.3, 0.0, 0.3, round(av_max, 2)})
        ax_av.set_yticks(yticks)
        ax_av.set_yticklabels([f"{v:g}" for v in yticks])

        # 强制在每一个角速度子图上都显示 y 轴刻度数值(不只是最左一列)
        ax_av.tick_params(axis="y", labelleft=True, labelsize=9)
        ax_av.tick_params(axis="x", labelsize=9)

        if col == 0:
            ax_av.set_ylabel("ang vel (rad/s)", fontsize=9)
        # 最底下一行 (block_row == n_rows-1) 才画 step 轴标签
        if block_row == n_rows - 1:
            ax_av.set_xlabel("step", fontsize=9)

    # 全局图例放在第一个轨迹子图里就够用了, 这里再加个总图例
    handles = [
        plt.Line2D([0], [0], marker="P", color="green", linestyle="None",
                   markersize=8, label="start (0, 0)"),
        plt.Line2D([0], [0], marker="*", color="red", linestyle="None",
                   markersize=10, label=f"goal ({target_distance}, 0)"),
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None",
                   markerfacecolor="none", markersize=7, label="end: success"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None",
                   markersize=7, label="end: collision"),
        plt.Line2D([0], [0], marker="s", color="black", linestyle="None",
                   markerfacecolor="none", markersize=7, label="end: timeout"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("", y=0.995)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] saved to {out_path}")


# ============================================================
# 结果保存
# ============================================================
def save_csv(rows, csv_path):
    if not csv_path:
        return
    # trajectory 不写进 CSV(可能很长), 想看就用 JSON
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
    all_rows = []                                # 每次测试一行(用于 CSV)
    summary_per_angle = []                       # 每个角度一项汇总
    avg_path_per_angle = []
    results_per_angle = []                       # 每个角度的完整 results(含 trajectory, 用于轨迹绘图)

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

        results_per_angle.append(per_angle_results)

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
    # CSV 不写 trajectory; JSON 里写一份精简版(每段轨迹保留 (x, y) list)
    save_csv(all_rows, CSV_PATH)

    json_results = []
    for runs in results_per_angle:
        for r in runs:
            json_results.append({
                **{k: v for k, v in r.items()
                   if k not in ("trajectory", "ang_vel_cmd")},
                "trajectory": [[round(x, 4), round(y, 4)]
                               for (x, y) in r["trajectory"]],
                "ang_vel_cmd": [round(v, 4) for v in r.get("ang_vel_cmd", [])],
            })

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
        "raw_results": json_results,
    }, JSON_PATH)

    # ---------- 出图 ----------
    # 顶部: path length vs angle (sweep 时才有意义), 下方: 每个角度的轨迹.
    # 单角度模式也仍然画一张轨迹图(顶部 path length 退化成单点).
    plot_combined(
        angle_list=angle_list,
        avg_path_lengths=avg_path_per_angle,
        results_per_angle=results_per_angle,
        target_distance=TARGET_DISTANCE,
        target_reached_delta=TARGET_REACHED_DELTA,
        runs_per_angle=RUNS_PER_ANGLE,
        out_path=PLOT_PATH,
        traj_max_cols=TRAJ_MAX_COLS,
    )

    # ---------- 收尾 ----------
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
