# from PIL import Image
# import numpy as np
# im = Image.open("C:/Users/Robert/Desktop/code1.bmp")
# p = np.array(im)
# p2 = []
#
# for i in range(25):
#     p2.append([])
#     for j in range(25):
#         p2[i] += [p[i, j, 0] % 254]
# p2 = np.asarray(p2)
# np.savetxt("C:/Users/Robert/Desktop/code1.csv", p2, delimiter=",", fmt="%d")

from os import listdir
from os.path import isfile, join
mypath = "C:\\Users\\Robert\\Documents\\Neo-Zero Soundtrack\\mp3\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import os

for f in onlyfiles:
    original = f
    f = f.split(" - ")
    f = f[2]
    os.rename(mypath + original, mypath + f)
