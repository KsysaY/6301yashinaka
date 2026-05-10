from concurrent.futures import ThreadPoolExecutor

def task(x):
    total = 0
    for _ in range(3):
        total += x
    return total

with ThreadPoolExecutor(max_workers=2) as ex:
    results = list(ex.map(task, [1, 2, 3]))
    print(sum(results))