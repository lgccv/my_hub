#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    camera_name_arg = DeclareLaunchArgument(
        "camera_name",
        default_value="camera",
        description="Orbbec camera namespace",
    )

    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/camera_02/color/image_raw",
        description="RGB image topic for person tracking",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera_02/depth/image_raw",
        description="Depth image topic for person tracking",
    )

    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")

    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("orbbec_camera"),
                "launch",
                "dabai_dcw2.launch.py",
            ])
        ),
        launch_arguments={
            "depth_fps" : "10",
            "ir_fps" : "10",
            "enable_ir" : "true",
            "depth_registration":"true",
            "camera_name":"camera_02",
            "enable_point_cloud":"false"
        }.items(),
    )

    person_track_launch = TimerAction(
        period=2.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("easygo_person_track"),
                        "launch",
                        "easygo_person_track.launch.py",
                    ])
                ),
                launch_arguments={
                    "rgb_topic": rgb_topic,
                    "depth_topic": depth_topic,
                }.items(),
            )
        ],
    )

    return LaunchDescription([
        camera_name_arg,
        rgb_topic_arg,
        depth_topic_arg,
        orbbec_camera_launch,
        person_track_launch,
    ])