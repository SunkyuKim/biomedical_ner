# import re
# s = "On the other hand factor IX activity is decreased in coumarin treatment with factor IX antigen remaining normal."
#
# offsets = [(14,21), (64,71)]
#
# tokens = s.split(' ')
#
#
# bio = [[0,0,1]]*len(tokens)
# for o in offsets:
#     s = 0
#     start_tag = None
#     end_tag = None
#     for i in range(len(tokens)):
#         if o[0] in range(s, s+len(tokens[i])+1):
#             start_tag = i
#         if o[1] in range(s, s + len(tokens[i]) + 1):
#             end_tag = i
#         s += len(tokens[i])
#
#
#     # print start_tag, end_tag
#     for i in range(start_tag, end_tag+1):
#         bio[i] = [0,1,0]
#
#     bio[start_tag] = [1,0,0]
# import numpy as np
# print np.array(bio)

import re

# a = '.1.2,'
# if a[0] in [',','.']:
#     a = a[1:]
# if a[-1] in [',','.']:
#     a = a[:-1]
# print a
#
# import numpy as np
#
#
# l = ['a', 'b', 'c', 'd', 'e']
# q = 'd'
# if q not in l:
#     l.append(q)
#     print len(l)-1
# else:
#     print l.index(q)


import numpy as np

batch_size = 2

t = [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12],
     [13,14,15,16],
      [17,18,19,20],
      [21,22,23,24],
     [25,26,27,28]
     ]


npt = np.array(t)

num_batches = int(npt.size / (batch_size * 4))
print num_batches
npt = npt.reshape(npt.size)
npt = npt[:num_batches * batch_size * 4]
print npt.shape

print np.split(npt.reshape(batch_size,-1), num_batches