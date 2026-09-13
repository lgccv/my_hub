import rclpy
from rclpy.node import Node
from learning_interface.srv import AddTwoInts
import debugpy

class adderServer(Node):
    def __init__(self,name):
        super().__init__(name)
        self.srv = self.create_service(AddTwoInts,"add_two_ints",self.adder_callback)


    def adder_callback(self,request,response):
        response.sum = request.a + request.b
        self.get_logger().info("Incoming request \na: %d b:%d" %(request.a,request.b))
        return response


def main(args=None):

    debugpy.listen(("0.0.0.0",5680))
    print("wait for debugger attach on port 5678.....")
    debugpy.wait_for_client()

    rclpy.init(args=args)
    node = adderServer("server_adder_server")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
        