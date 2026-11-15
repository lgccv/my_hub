# 进程内的消息发送

from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Callable

@dataclass
class ObstacleMsg:
    topic: str="OBSTACLES"
    oba_name: str = ""
    point_cloud: list[tuple[float,float,float]] = field(default_factory=list)


class MsgBus:
    def __init__(self):
        self.queue = Queue()
        self.subscribers: dict[str, list[Callable]] = {}
        self.running = True

        self.thread = Thread(target= self.dispath,daemon=True)
        self.thread.start()

    def subscribe(self,topic:str,callback:Callable):
        self.subscribers.setdefault(topic,[]).append(callback)

    def send_msg(self, msg):
        self.queue.put(msg)

    def dispathch(self):
        while self.running:
            msg = self.queue.get()

            # 根据 topic 找到所有订阅者
            callbacks = self.subscribers.get(msg.topic,[])
            for callback in callbacks:
                callback(msg)

    def stop(self):
        self.running = False

def obstacle_callback(msg: ObstacleMsg):
    print("收到消息")
    print("topic:",msg.topic)
    print("oba_name:",msg.oba_name)
    print("points:",msg.point_cloud)


# 创建消息总线
bus = MsgBus()

bus.subscribe("OBSTACLES",obstacle_callback)

oba_msg = ObstacleMsg(topic="OBSTACLES",oba_name="obstacle_camera_1",point_cloud=[(1.0,0.2,0.1),(1.2,0.3,0.1)])


# 相当于
bus.send_msg(oba_msg)