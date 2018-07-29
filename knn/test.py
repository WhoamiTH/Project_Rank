import numpy as np
import heapq
import random

# a = np.array([1, 100, 3, 30, 5])
# b = [1,4]
# t = a.take(b)
# print(type(t))
# print(t)
# x= t.tolist()
# print(type(t))
# print(type(x))
# t = t.tolist()
# print(type(t))
# t = heapq.nlargest(3, range(len(a)), a.take)
# print(t)

# def l1(alist):
#     for each in range(len(alist)):
#         alist[each] += 100
#         print(alist[each])
#     print(alist)


# def l1(alist):
#     for each in range(10):
#         alist.append(each)
#     print(alist)
#
# def l2(alist):
#     l1(alist)
    # for each in range(10):
    #     alist.append(each)
    # print(alist)

#
# alist = [i for i in range(10)]
# l2(alist)
# print(alist)

l = [i for i in range(10)]
t = [x for x in range(10)]
l+=t
print(l)