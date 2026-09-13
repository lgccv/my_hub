import sys
import rclpy

from rclpy.node import Node
from learning_interface.srv import AddTwoInts
import debugpy

class adderClient(Node):
    def __init__(self,name):
        super().__init__(name)
        self.client = self.create_client(AddTwoInts,"add_two_ints")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available,waiting again....")
        self.request = AddTwoInts.Request()

    def send_request(self):
        self.request.a = int(sys.argv[1])
        self.request.b = int(sys.argv[2])
        self.future = self.client.call_async(self.request)
        print("result",self.future)

def main(args=None):

    debugpy.listen(("0.0.0.0",5682))
    print("wait for debugger attach on port 5678.....")
    debugpy.wait_for_client()


    rclpy.init(args=args)
    node =  adderClient("service_adder_client")
    node.send_request()


    while rclpy.ok():
        rclpy.spin_once(node)

        if node.future.done():
            try:
                response = node.future.result()
            except Exception as e:
                node.get_logger().info("Service call failed %r" %(e,))
            else:
                node.get_logger().info("Result of add_two_ints: for %d + %d =%d" % 
                                       (node.request.a,node.request.b,response.sum))
            break
    node.destroy_node()
    rclpy.shutdown()