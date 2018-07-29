import numpy as np
import heapq

# a = np.array([1, 100, 3, 30, 5])
# t = heapq.nlargest(3, range(len(a)), a.take)
# print(t)

def li(alist):
    for each in range(len(alist)):
        alist[each] += 100
        print(alist[each])
    print(alist)


alist = [i for i in range(10)]
t = li(alist)
print(alist)
print(t)