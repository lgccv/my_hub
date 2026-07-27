import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from learning_interface.action import MoveCircle
import debugpy

class MoveCircleActionServer(Node):
    def __init__(self,name):
        super().__init__(name)
        self._action_server = ActionServer(self,MoveCircle,"move_circle",self.execute_callback)

    def execute_callback(self,goal_handle):  # 执行收到动作目标之后的处理函数
        self.get_logger().info("Moving circle.......")
        feedback_msg = MoveCircle.Feedback()

        for i in range(0,360,30):
            feedback_msg.state = i
            self.get_logger().info("Publishing feedback:%d" % feedback_msg.state)
            goal_handle.publish_feedback(feedback_msg)  # 反馈过程
            time.sleep(0.5)

        goal_handle.succeed()
        result = MoveCircle.Result()
        result.finish = True
        return result   # 反馈最终动作执行的结果

def main(args=None):
    debugpy.listen(("0.0.0.0",5683))
    print("wait for debugger attach on port 5678.....")
    debugpy.wait_for_client()
    rclpy.init(args=args)
    node = MoveCircleActionServer("aciton_move_server")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

        