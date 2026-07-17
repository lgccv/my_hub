from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import parameter
from ament_index_python.packages import get_package_share_directory
# def generate_launch_description():
#     return LaunchDescription([
#         Node(package="easygo_person_track",
#              executable = "person_track_node",
#              name = "easygo_person_track_node",
#              output="screen",
#              parameters=[
#                 {
#                     "rgb_topic": "rgb_image",
#                     "depth_topic": "depth_image",
#                     "tracked_target_topic": "tracked_target",
#                     "action_server_name": "start_binding",
#                     "model_path": "./model/last_quant.rknn",
#                     "det_conf_threshold": 0.25,
#                     "det_nms_threshold": 0.45,
#                 }],
#             )])


def generate_launch_description():
    default_model_path = os.path.join(
        get_package_share_directory("easygo_person_track"),
        "model",
        "last_quant.rknn",
    )
    args = [
        DeclareLaunchArgument('rgb_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('tracked_target_topic', default_value='/follow/perception/tracked_target'),
        DeclareLaunchArgument('action_server_name', default_value='/follow/perception/start_binding'),
        DeclareLaunchArgument('model_path', default_value=default_model_path),
        DeclareLaunchArgument('det_conf_threshold', default_value="0.25"),
        DeclareLaunchArgument('det_nms_threshold', default_value="0.45"),
    ]

    parameters = [{
    "rgb_topic": LaunchConfiguration("rgb_topic"),
    "depth_topic": LaunchConfiguration("depth_topic"),
    "tracked_target_topic": LaunchConfiguration("tracked_target_topic"),
    "action_server_name": LaunchConfiguration("action_server_name"),
    "model_path": LaunchConfiguration("model_path"),
    "det_conf_threshold": ParameterValue(
        LaunchConfiguration("det_conf_threshold"),
        value_type=float,
    ),
    "det_nms_threshold": ParameterValue(
        LaunchConfiguration("det_nms_threshold"),
        value_type=float,
    ),
    }]

    return LaunchDescription(
            args
            + [Node(
                    package="easygo_person_track",
                    executable="person_track_node",
                    name="easygo_person_track_node",
                    output="screen",
                    parameters=parameters
                )
            ]
        )

