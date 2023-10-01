from queue import Queue, Empty
from typing import TypeVar, Callable
from threading import Thread

T = TypeVar('T')
U = TypeVar('U')

class ProcessingContext:
    """
    A context for processing items in parallel.
    """
    def __init__(self, items, process):
        self.queue = Queue(maxsize=len(items))
        """
        A queue of indicies pointing to the items collection.
        """

        self.items = items
        """
        An indexable collection of items.
        """

        self.process = process
        """
        Function that performs processing.
        """

        self.results = [None] * len(items)
        """
        Contains result of each item. (Results may be exceptions.)
        """

        for index in range(len(items)):
            self.queue.put(index)

def __internal_process_all(context:ProcessingContext):
    """
    An internal function.

    Processes all items in the context.
    """
    try:
        while context.queue.qsize() != 0:
            index = context.queue.get()

            try:
                item = context.items[index]
                result = context.process(item)
                context.results[index] = result
            except Exception as error:
                context.results[index] = error
    except Empty:
        pass
    except Exception as e:
        print(e)


def parallel_map(mapping:Callable[[T], U], items:list[T], thread_count=5) -> list[U]:
    """
    Maps items by using the provided mapping. 
    The mapping is done in parallel on the given amount of threads.

    Returns:
    The mapped list. Order is preserved.
    """
    context = ProcessingContext(items, mapping)
    
    threads: list[Thread] = []

    for _ in range(thread_count):
        thread = Thread(target=__internal_process_all, args=(context,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return context.results