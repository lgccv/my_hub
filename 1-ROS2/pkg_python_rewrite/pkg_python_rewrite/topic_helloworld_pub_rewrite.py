
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import debugpy

class PublisherNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.pub = self.create_publisher(String,"chatter",10)
        self.timer = self.create_timer(0.5,self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = "Hello World"
        self.pub.publish(msg)
        self.get_logger().info('Publishing: "%s")' % msg.data)


def main(args=None):

    # debugpy.listen(("0.0.0.0",5678))
    # print("wait for debugger attach on port 5678.....")
    # debugpy.wait_for_client()

    rclpy.init(args=args)
    node = PublisherNode("topic_helloworld_pub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
        