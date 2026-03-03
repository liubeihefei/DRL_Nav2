import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]


def generate_launch_description():
    # 使用模拟时间（可能是gazebo与现实时间不同步）
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    # Gazebo是否暂停，可能用于调试
    pause = LaunchConfiguration("pause", default="false")
    # 加载世界文件
    world_file_name = "turtlebot3_drl/" + TURTLEBOT3_MODEL + ".model"
    world = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "worlds", world_file_name
    )
    # 获取launch文件目录，用来启动另一个launch文件
    launch_file_dir = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "launch"
    )
    # 获取gazebo_ros包的路径，用来启动gazebo相关的launch文件，但源码里面就没有这个包，应该是去调用gazebo的工具包了
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    return LaunchDescription(
        [
            # gazebo服务器
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
                ),
                launch_arguments={"world": world, "pause": pause}.items(),
            ),
            # gazebo客户端，显示仿真环境，注释后不显示，可无头训练
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(
            #         os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
            #     ),
            # ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [launch_file_dir, "/robot_state_publisher.launch.py"]
                ),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
        ]
    )
