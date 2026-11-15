from dataclasses import dataclass
from threading import Event, Lock, Thread


# ============================================================
# 1. 错误示例：直接修改共享对象
# ============================================================

class MutableRange:
    def __init__(self, min_x, max_x):
        self.min_x = min_x
        self.max_x = max_x


class UnsafeExample:
    def __init__(self):
        self.ranges = MutableRange(0.0, 1.0)

        # 控制线程执行顺序
        self.half_written = Event()
        self.allow_finish = Event()

    def writer(self):
        # 先修改第一个字段
        self.ranges.min_x = 10.0

        # 通知读取线程：现在只写了一半
        self.half_written.set()

        # 等待读取线程完成读取
        self.allow_finish.wait()

        # 再修改第二个字段
        self.ranges.max_x = 20.0

    def reader(self):
        # 等待写线程只修改完 min_x
        self.half_written.wait()

        # 此时读取到的是混合状态：
        # min_x 已经是新值，max_x 还是旧值
        print(
            "[错误示例] 读取到:",
            f"min_x={self.ranges.min_x},",
            f"max_x={self.ranges.max_x}",
        )

        self.allow_finish.set()


unsafe = UnsafeExample()

writer_thread = Thread(target=unsafe.writer)
reader_thread = Thread(target=unsafe.reader)

writer_thread.start()
reader_thread.start()

writer_thread.join()
reader_thread.join()


# ============================================================
# 2. 正确示例：先构造完整对象，再一次性替换
# ============================================================

@dataclass(frozen=True)
class SpaceRange3D:
    min_x: float
    max_x: float


class AtomicSnapshot:
    def __init__(self, value):
        self._value = value
        self._lock = Lock()

    def store(self, value):
        # 类似 C++:
        # std::atomic_store(&ranges_, value);
        with self._lock:
            self._value = value

    def load(self):
        # 类似 C++:
        # std::atomic_load(&ranges_);
        with self._lock:
            return self._value


class SafeExample:
    def __init__(self):
        self.ranges = AtomicSnapshot(
            SpaceRange3D(
                min_x=0.0,
                max_x=1.0,
            )
        )

        self.ready_to_store = Event()
        self.allow_store = Event()

    def writer(self):
        # 先在本地构造完整的新对象
        new_ranges = SpaceRange3D(
            min_x=10.0,
            max_x=20.0,
        )

        # 通知读取线程：新对象已经构造完成，
        # 但还没有发布
        self.ready_to_store.set()

        # 等待读取线程读取旧快照
        self.allow_store.wait()

        # 一次性替换整个对象
        self.ranges.store(new_ranges)

    def reader(self):
        # 等待写线程准备好新对象，但尚未 store
        self.ready_to_store.wait()

        # 只能读到完整的旧对象，
        # 不会读到半新半旧的数据
        old_ranges = self.ranges.load()

        print(
            "[正确示例] store 前读取:",
            f"min_x={old_ranges.min_x},",
            f"max_x={old_ranges.max_x}",
        )

        self.allow_store.set()

        # 等写线程完成一次性替换
        updated_ranges = self.ranges.load()

        print(
            "[正确示例] store 后读取:",
            f"min_x={updated_ranges.min_x},",
            f"max_x={updated_ranges.max_x}",
        )


safe = SafeExample()

writer_thread = Thread(target=safe.writer)
reader_thread = Thread(target=safe.reader)

writer_thread.start()
reader_thread.start()

writer_thread.join()
reader_thread.join()