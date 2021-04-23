from multiprocessing import Queue, Process
from WiPHY.utils import crc_remainder

def write_to_queue(queue):
    for i in range(0, 10**6):
        queue.put(i)
        print("Current queue size: %d"%(queue.qsize()))

def read_from_queue(queue):
    while True:
        msg = queue.get()
    
if __name__ == "__main__":
    """
    pqueue = Queue(maxsize=1000)
    target_functions = [write_to_queue, read_from_queue]
    processes = []
    for func in target_functions:
        process = Process(target=func, args=((pqueue), ))
        process.daemon = True
        process.start()
        processes.append(process)
    
    for p in processes:
        p.join()
    print("Done scheduling.")
    """
    print(crc_remainder("01111111", "1101", "0"))
