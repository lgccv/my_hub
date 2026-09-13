from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    action_client = Node(
                        package="pkg_python_rewrite",   # 节点所在功能包
                        executable="action_move_client_rewrite",  # 节点的可执行文件名
                        namespace="action",   # 节点所在的命名空间
                        name = "sim",         # 对节点重新命名
                        remappings=[
                                    ('/input/pose', '/turtlesim1/turtle1/pose'),         # 将/input/pose话题名修改为/turtlesim1/turtle1/pose
                                    ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),  # 将/output/cmd_vel话题名修改为/turtlesim2/turtle1/cmd_vel
                        ])             # 对节点重新命名

    action_server = Node(
                        package="pkg_python_rewrite",
                        executable="action_move_server_rewrite",
    )

    server_server = Node(
                        package="pkg_python_rewrite",
                        executable="service_adder_server_rewrite",
    )

    server_client = Node(
                        package="pkg_python_rewrite",
                        executable="service_adder_client_rewrite",
                        arguments=["1","2"]
    )

    topic_pub = Node(
                        package="pkg_python_rewrite",
                        executable="topic_helloworld_pub",
                        namespace="kkkkkk",
    )

    topic_sub = Node(
                        package="pkg_python_rewrite",
                        executable="topic_helloworld_sub",
    )
    

    return LaunchDescription([action_client,action_server,server_server,server_client,topic_pub,topic_sub])