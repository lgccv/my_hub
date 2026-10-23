import time
import grpc
import hello_bilibili_pb2 as pb2
import hello_bilibili_pb2_grpc as pb2_grpc
from concurrent import futures


class Bilili(pb2_grpc.BililiServicer): # 服务名要和protobuf保持一致
    def HelloDewei(self, request, context):  # 必须要有同名方法
        name = request.name
        age = request.age

        result = f"my name is {name} , i am {age} years old"
        return pb2.HelloDeweiReply(result=result)  # 返回同名方法


def run():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_BililiServicer_to_server(Bilili(), grpc_server)
    grpc_server.add_insecure_port("0.0.0.0:50051")
    grpc_server.start()
    print("gRPC server started on 0.0.0.0:50051")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)



if __name__ == "__main__":
    run()





# 环境安装:
# pip install grpcio grpcio-tools
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. hello_bilibili.proto