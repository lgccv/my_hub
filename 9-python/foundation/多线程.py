import threading
from concurrent.futures import ThreadPoolExecutor

lock_object = threading.RLock()


def _add(count):
    lock_object.acquire()    # 申请锁
    name = threading.current_thread().getName()
    print("hhhhh")
    lock_object.release()    # 释放锁

t1 = threading.Thread(target=_add)
t1.setName("name1")
t1.start()

t1.join()  # t1线程执行完毕，才继续往后走



pool = ThreadPoolExecutor(10)

future_list = []

for i in range(10):
    future = pool.submit()
    future_list.append(future)


pool.shutdown(True) 