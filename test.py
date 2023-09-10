from collections import Counter
import random
import torch
from sklearn.metrics import classification_report

# l = [1, 4, 6, 4, 4, 6, 2]
# newl = []
# transdict = {}
# j = 0
# for i in l:
#     print(newl)
#     if i not in transdict:
#         print(i)
#         transdict[i] = j
#         j += 1
#         newl.append(transdict[i])
#     else:
#         newl.append(transdict[i])

# print(Counter(l))
# print(type(dict(Counter(l))))
# print(dict(Counter(l)))

# while len(newl) < 9:
#     count = dict(Counter(newl))
#     min_keys = [k for k, v, in count.items() if v == min(count.values())]
#     add_key = random.choice(min_keys)
#     newl.append(add_key)

# print(newl)

# def test(a):
#     return a, a+1

# i = 1
# s = test(i)
# print(s)

l = [0, 0, 1]
l2 = [0, 1, 0]
# p = classification_report(l, l2)

# print(p[30])
# print(type(p))
l3 = [0, 0, 1, 2, 2, 2, 3, 1, 3]
# r = [2, 2, 2, 2, 2, 2, 2, 2]
# r2 = []

def GetBalancePosNegPair(ip_labels):
        # panos = []
        labels = []
        count = dict(Counter(labels))
        max_label = max(ip_labels)
        label_cnt = 0

        # random get 2
        while (label_cnt < max_label):
            if (count.get(label_cnt) == None):
                cur_label_keys = [k for k, v in enumerate(ip_labels) if v == label_cnt]
                print(cur_label_keys)
                cur_get_labels = random.sample(cur_label_keys, 2)
                print(cur_get_labels)
                for j in cur_get_labels:
                    # panos.append(ip_panos[j])
                    labels.append(ip_labels[j])
                label_cnt += 1

        return labels

# l4 = GetBalancePosNegPair(l3)
# print(l4)

# count = dict(Counter(l3))
# print(len(count.items()))

l5 = [(0, 1), (1, 3), (8, 6), (5, 2), (9, 4)]
l5 = sorted(l5, key = lambda x: x[1])
print(l5)

# def d(l):
#     count = dict(Counter(l))
#     count2 = dict(Counter(r2))
#     for label in l:
#         c = 0 
#         if count2.get(label):
#             c = (count2[label])
#         print(c)
#         if c < 2:
#             r2.append(label)
#         count2 = dict(Counter(r2))

# def c(l):
#     count = dict(Counter(l))
#     for idx, label in enumerate(l):
#         print(idx, label)
#         if (count[label] > 2):
#             l.pop(idx)
#             # panos.pop(idx)
#             count = dict(Counter(l))
# # c(l)
# # c(r)
# d(l)
# # d(r)
# print(r2)
# print(l)

# l = []
# if l:
#     print("y")
# else:
#     print("N")

# c = 0
# c += 1 == 0
# print(c)

# a = 3
# print(a/2)
# print(a/2.0)
# print(a//2)

# b = [[1,2], [3,4], [5,6]]
# c = []
# c += b[a//2:]
# print(c)
# a = torch.tensor([6, 5, 6])
# print(a)
# print(torch.where(a == torch.max(a))[0].tolist()[1])
# [
#     ( tensor(panos), tensor(labels) )
# ]