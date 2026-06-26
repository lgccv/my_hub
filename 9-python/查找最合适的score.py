import csv
import json
import sys
import time
import os
import datetime
import cv2
import shutil
import random
from tqdm import tqdm
import numpy as np





# 1、numpy的排序
# 2、numpy的切片(按某个值切片)
# 3、numpy找最大值


ll = [[1,2,3,0.1],[4,5,6,0.2]]
output_data = np.array(ll)
print(output_data)
print('~'*50)



# numpy的排序
ScoreSort = np.argsort(output_data[:, 3])[::-1]
print(ScoreSort)
print(ScoreSort[::-1])
det = output_data[ScoreSort]
print(det)
exit()

scores = np.max(output_data[1], 2)
labels = np.argmax(output_data[1], 2)

# ValIdxs = det[:, 4] 
# det = det[ValIdxs]
ScoreSort = np.argsort(det[:, 4])[::-1]
det = det[ScoreSort]

np.sort()



