# 这个例程的重点是显示两个线程访问同一个变量，一个赋值和一个取值，要赋玩值才能取值

from dataclasses import dataclass
from threading import Lock, Thread
import time


@dataclass(frozen=True)
class SpaceRange3D:
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0


class AtomicSnapshot:
    def __init__(self, initial_value):
        self._value = initial_value
        self._lock = Lock()

    def store(self, new_value):
        # 类似 C++ 的 std::atomic_store(&ranges_, new_value)
        with self._lock:
            self._value = new_value

    def load(self):
        # 类似 C++ 的 std::atomic_load(&ranges_)
        with self._lock:
            return self._value


class PointCloudProcess:
    def __init__(self):
        self.ranges = AtomicSnapshot(
            SpaceRange3D(
                min_x=0.0,
                max_x=1.0,
                min_y=-1.0,
                max_y=1.0,
            )
        )

    def update_ranges_loop(self):
        for i in range(5):
            # 先创建完整的新快照
            new_ranges = SpaceRange3D(
                min_x=float(i),
                max_x=float(i + 1),
                min_y=-1.0,
                max_y=1.0,
            )

            # 一次性替换
            self.ranges.store(new_ranges)

            print("[更新线程] 发布配置:", new_ranges)
            time.sleep(0.1)

    def process_points_loop(self):
        for _ in range(10):
            # 先取得一个稳定快照
            current_ranges = self.ranges.load()

            # 后续处理期间使用这个快照
            print("[处理线程] 使用配置:", current_ranges)

            time.sleep(0.05)


process = PointCloudProcess()

update_thread = Thread(target=process.update_ranges_loop)
process_thread = Thread(target=process.process_points_loop)

update_thread.start()
process_thread.start()

update_thread.join()
process_thread.join()