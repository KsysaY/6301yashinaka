import time

#Декоратор, замеряющий время выполнения функции    
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} - {end - start:.4f} сек")
        return result
    return wrapper