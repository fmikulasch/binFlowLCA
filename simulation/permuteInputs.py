from __future__ import print_function
from random import shuffle

path = "../testInput/"
imageNames = [[open(path + 'inputImagesLeft' +
                    str(i) + '.txt', 'a+') for i in xrange(4)],
              [open(path + 'inputImagesRight' +
                    str(i) + '.txt', 'a+') for i in xrange(4)]]

n = 9
lines = [[[f.readline()[:-1] for f in imageNames[i]] for i in xrange(2)]
         for j in xrange(n)]


for p in xrange(4):
    r = [i for i in range(1,n)]
    shuffle(r)

    for i in r:
        print(i, end=" ")
        for s in xrange(2):
            for axis in xrange(4):
                print(lines[i][s][axis], file=imageNames[s][axis])
