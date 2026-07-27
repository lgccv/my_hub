import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import debugpy


class SubscriberNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.sub = self.create_subscription(String,"chatter",self.listener_callback,10)

    def listener_callback(self,msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    debugpy.listen(("0.0.0.0",5679))
    print("wait for debugger attach on port 5678.....")
    debugpy.wait_for_client()

    rclpy.init(args=args)
    node = SubscriberNode("topic_hellowold_sub")
    rclpy.spin(node)  # 循环等待ROS2退出
    node.destroy_node()
    rclpy.shutdown()