from queue import Queue
from threading import Thread
import time


class Message:
    def __init__(self, topic, data=None):
        self.topic = topic
        self.data = data


class MsgBus:
    def __init__(self):
        self.main_queue = Queue()
        self.topic_modules = {}
        self.running = True

    def registerTopic(self, topic, module):
        modules = self.topic_modules.setdefault(topic, [])
        if module not in modules:
            modules.append(module)

    def sendMsg(self, msg):
        self.main_queue.put(msg)

    def dispatch(self):
        while self.running:
            msg = self.main_queue.get()

            # 一个主题可以有多个订阅模块
            for module in self.topic_modules.get(msg.topic, []):
                module._putMessage(msg)

    def start(self):
        Thread(target=self.dispatch, daemon=True).start()


class Module:
    def __init__(self, name, bus):
        self.name = name
        self.bus = bus
        self.self_queue = Queue()
        self.callbacks = {}

    def sendMsg(self, msg):
        # 等价于 C++ Module::sendMsg(msg)
        self.bus.sendMsg(msg)

    def subscribeTopic(self, topic, callback, max_execution_time_ms=200):
        # 等价于 C++ Module::subscribeTopic(...)
        self.callbacks[topic] = callback
        self.bus.registerTopic(topic, self)

    def _putMessage(self, msg):
        self.self_queue.put(msg)

    def dispatch(self):
        while True:
            msg = self.self_queue.get()
            callback = self.callbacks.get(msg.topic)

            if callback is not None:
                callback(msg)

    def start(self):
        Thread(target=self.run, daemon=True).start()

    def run(self):
        raise NotImplementedError


# ============================================================
# 示例一：模仿 two_piers_pcl_detection.cpp
# ============================================================

class DetectorModuleBase:
    """
    对应 C++ 的 DetectorModuleBase。

    它自己不是全局 Module，而是通过构造函数接收：
    - sendMsg
    - subscribeTopic
    """

    def __init__(self, sendMsg, subscribeTopic):
        self.sendMsg_ = sendMsg
        self.subscribeTopic_ = subscribeTopic

        self.enable_process_ = False
        self.is_init_detector_ = True

    def startDetector(self):
        self.enable_process_ = True

    def stopDetector(self):
        self.enable_process_ = False


class TwoPiersPclDetectModule(DetectorModuleBase):
    def __init__(self, sendMsg, subscribeTopic):
        super().__init__(
            sendMsg=sendMsg,
            subscribeTopic=subscribeTopic,
        )

        self.detector_mode_ = 0
        self.command_ = None

    def onDetectCommandMsg(self, msg):
        if not self.is_init_detector_:
            print("[TwoPiers] detector disabled")
            return

        cmd = msg.data
        self.detector_mode_ = cmd["detector_mode"]
        self.command_ = cmd

        print(
            "[TwoPiers] receive command:",
            "is_enable=", cmd["is_enable"],
            "detector_mode=", cmd["detector_mode"],
        )

        if cmd["is_enable"]:
            # 完全对应 C++：
            #
            # subscribeTopic_(
            #     "DEPTH_CAMERA_ORIGIN",
            #     CALLBACK(&TwoPiersPclDetectModule::onSensorMsg));
            #
            # Python 的 self.onSensorMsg 已经自动绑定了 self，
            # 不需要 CALLBACK 宏。

            self.subscribeTopic_(
                "DEPTH_CAMERA_ORIGIN",
                self.onSensorMsg,
            )

            self.subscribeTopic_(
                "TIMER_50MS",
                self.checkDetectTime,
            )

            self.startDetector()

        else:
            if self.enable_process_:
                self.stopDetector()

    def onSensorMsg(self, msg):
        if not self.enable_process_:
            return

        print(
            "[TwoPiers] receive DEPTH_CAMERA_ORIGIN:",
            msg.data,
        )

    def checkDetectTime(self, msg):
        print("[TwoPiers] check detect timeout")


class PerceptionManager(Module):
    """
    模拟 C++ 的 PerceptionManager。

    TwoPiersPclDetectModule 的 subscribeTopic_ 实际上会回调到这里，
    然后由 PerceptionManager 自己调用 Module::subscribeTopic。
    """

    def __init__(self, bus):
        super().__init__("PerceptionManager", bus)

        self.two_piers_detector = TwoPiersPclDetectModule(
            sendMsg=self.massageCallback,
            subscribeTopic=self.subscribeTopicCallback,
        )

    def massageCallback(self, msg):
        # 对应 C++：
        # void PerceptionManager::massageCallback(...) {
        #     sendMsg(m);
        # }
        self.sendMsg(msg)

    def subscribeTopicCallback(self, topic, callback):
        # 对应 C++：
        # void PerceptionManager::subscribeTopicCallback(...) {
        #     subscribeTopic(topic, process);
        # }
        self.subscribeTopic(topic, callback)

    def run(self):
        self.subscribeTopic(
            "DETECT_COMMAND",
            self.onDetectCommandMsg,
        )

        self.dispatch()

    def onDetectCommandMsg(self, msg):
        # PerceptionManager 收到命令后转发给检测器
        self.two_piers_detector.onDetectCommandMsg(msg)


# ============================================================
# 示例二：模仿 point_cloud_process_module.cpp
# ============================================================

class PointCloudProcessModule(Module):
    """
    直接继承 Module，在 run() 中自己订阅主题。
    """

    def __init__(self, bus):
        super().__init__("PointProcess", bus)

    def run(self):
        # 对应 C++ PointCloudProcessModule::run()

        self.subscribeTopic(
            "TIMER_50MS",
            self.updateParaLoop,
            50,
        )

        self.subscribeTopic(
            "DEPTH_CAMERA_ORIGIN",
            self.onProcessObstacleCameraMsg,
            100,
        )

        self.dispatch()

    def updateParaLoop(self, msg):
        print("[PointCloudProcess] update parameters")

    def onProcessObstacleCameraMsg(self, msg):
        print(
            "[PointCloudProcess] receive DEPTH_CAMERA_ORIGIN:",
            msg.data,
        )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    bus = MsgBus()
    bus.start()

    perception_manager = PerceptionManager(bus)
    point_cloud_process = PointCloudProcessModule(bus)

    # 启动两个真正的 Module
    perception_manager.start()
    point_cloud_process.start()

    # 等待两个模块完成 run() 中的订阅
    time.sleep(0.1)

    # 发送检测启用命令
    bus.sendMsg(
        Message(
            "DETECT_COMMAND",
            {
                "is_enable": True,
                "detector_mode": 1,
            },
        )
    )

    time.sleep(0.1)

    # 发布一帧深度相机点云
    # 此时：
    # 1. TwoPiersPclDetectModule 会收到
    # 2. PointCloudProcessModule 也会收到
    bus.sendMsg(
        Message(
            "DEPTH_CAMERA_ORIGIN",
            {
                "sensor_name": "camera_1",
                "points": [
                    (1.0, 0.2, 0.1),
                    (1.2, 0.3, 0.1),
                ],
            },
        )
    )

    time.sleep(0.5)




    # python3 demo.py