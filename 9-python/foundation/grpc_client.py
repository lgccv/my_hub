import grpc
import hello_bilibili_pb2 as pb2
import hello_bilibili_pb2_grpc as pb2_grpc


def run():
    conn = grpc.insecure_channel("127.0.0.1:50051")
    client = pb2_grpc.BililiStub(channel=conn)
    response = client.HelloDewei(pb2.HelloDeweiReq(  # 服务请求要一致
        name="dewei",
        age=33,
    ))
    print(response.result)

if __name__ == "__main__":
    run()
