# def dijkstra(matrix, source):
#     """迪杰斯特拉算法实现
#     Args:
#         matrix (_type_): 用邻接矩阵表示带权图
#         source (_type_): 起点

#     Returns:
#         _type_: 最短路径的节点集合，最短路径的节点的最短距离，每个节点到起点的最短路径
#     """
#     INF = float('inf')
#     n = len(matrix)
#     m = len(matrix[0])
#     assert n == m, "Error, please examine matrix dim"
#     assert source < n, "Error, start point should be in the range!"

#     S = [source]
#     U = [v for v in range(n) if v not in S]
#     distance = [INF]*n
#     distance[source] =0
#     path_optimal = [[]]*n
#     path_optimal[source] = [source]
#     while len(S) < n:
#         min_value = INF
#         col =-1 
#         row =-1
#         for s in S:
#             for u in U:
#                 if matrix[s][u] + distance[s] < min_value:
#                     min_value = matrix[s][u] + distance[s]
#                     row =s
#                     col =u

#         if col == -1 or row==-1:
#             break

#         S.append(col)
#         U.remove(col)
#         distance[col] = min_value
#         path_optimal[col] = path_optimal[row][:]
#         path_optimal[col].append(col)

#     return S,distance,path_optimal



# def main():
#     INF = float('inf')

#     matrix = [[0,12,INF,INF,INF,16,14],
#               [12,0,10,INF,INF,7,INF],
#               [INF,10,0,3,5,6,INF],
#               [INF,INF,3,0,4,INF,INF],
#               [INF,INF,5,4,0,2,8],
#               [16,7,6,INF,2,0,9],
#               [14,INF,INF,INF,8,9,0]]
    
#     S, distance,path_optimal = dijkstra(matrix,3)

#     print('S:')
#     print(S)
#     print('distance:')
#     print(distance)
#     print('path_optimal:')
#     for p in path_optimal:
#         print(p)


# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import copy

import sys
sys.path.append("/Users/jodocls/Desktop/code/my_hub/0-路径规划/chhRobotics/PathPlanning/Dijkstra/demo")

import map

def Dijkstra_NextNode(field,node):
    rows = len(field)
    cols = len(field[0])

    movepos = [[-1,1],[0,1],[1,1],[-1,0],[1,0],[-1,-1],[0,-1],[1,-1]]

    nextnodes = [[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf],[np.inf,np.inf]]

    nodeSub = map.ind2sub([rows,cols],node)

    for i in range(8):
        r = nodeSub[0] + movepos[i][0]
        c = nodeSub[1] + movepos[i][1]

        # 父节点有效移动范围,不能越界
        if -1 < r and r < rows and -1 <c and c<cols:
            nextnodes[i][0] = map.sub2ind([rows,cols],r,c)

            if field[r,c] !=2:
                dist = np.sqrt(movepos[i][0]*movepos[i][0]+movepos[i][1]*movepos[i][1])
                nextnodes[i][1] = dist

    return nextnodes

# 初始化地图
rows = 5
cols = 6
startSub = [2,1]
goalSub = [2,5]
obsSub = [[1,3],[2,3],[3,3]]


# 栅格地图属性
field = np.ones((rows,cols))

field[startSub[0],startSub[1]] =4
field[goalSub[0],goalSub[1]] =5

for i in range(len(obsSub)):
    field[obsSub[i][0],obsSub[i][1]] =2


# 数据转换
startIndex = map.sub2ind([rows,cols],startSub[0],startSub[1])
goalIndex = map.sub2ind([rows,cols],goalSub[0],goalSub[1])

'''
Dijkstra算法
'''
U_pos = []
U_dist = []


for i in range(rows*cols):
    U_pos.append(i)
    U_dist.append(np.inf)

S_pos = [startIndex]
S_dist = [0]

# 在U集合中删除起点的信息
idx = U_pos.index(startIndex)
U_pos.pop(idx)
U_dist.pop(idx)

path = []

for i in range(rows*cols):
    path.append([-1])

nextNodes = Dijkstra_NextNode(field,startIndex)

for i in range(8):
    nextnode = nextNodes[i][0]

    if nextnode != np.inf:
        idx = U_pos.index(nextnode)
        U_dist[idx] = nextNodes[i][1]


    if nextNodes[i][1] != np.inf:
        path[nextnode] = copy.deepcopy([startIndex,nextNodes[i][0]])

searhx = []
searhy = []
fig = plt.figure(figsize=(4,3))

ax = fig.add_subplot(111)
startXY =map.sub2xy([rows,cols],startSub[0],startSub[1])
goalXY = map.sub2xy([rows,cols],goalSub[0],goalSub[1])
obsX = []
obsY = []

for i in range(len(obsSub)):
    obsxy = map.sub2xy([rows,cols],obsSub[i][0],obsSub[i][1])
    obsX.append(obsxy[0])
    obsY.append(obsxy[1])


while len(U_pos) > 0:
    idx = U_dist.index(min(U_dist))
    dist_min = U_dist[idx]
    node = U_pos[idx]

    S_pos.append(node)
    S_dist.append(dist_min)

    U_pos.pop(idx)
    U_dist.pop(idx)

    # 绘图1
    nodesub = map.ind2sub([rows,cols],node)
    if field[nodesub[0]][nodesub[1]] ==1:
        nodexy = map.sub2xy([rows,cols],nodesub[0],nodesub[1])
        searhx.append(nodexy[0])
        searhy.append(nodexy[1])

    nextNodes = Dijkstra_NextNode(field,node)

    for i in range(8):
        nextnode = nextNodes[i][0]

        if nextnode != np.inf:
            if nextnode not in S_pos:
                idx_u = U_pos.index(nextnode)
                cost = nextNodes[i][1]

                if dist_min+cost < U_dist[idx_u]:
                    U_dist[idx_u] = dist_min + cost
                    path[nextnode] = copy.deepcopy(path[node])
                    path[nextnode].append(nextnode)

    plt.cla()
    plt.plot(startXY[0],startXY[1],"r+")
    plt.plot(goalXY[0],goalXY[1],'b+')
    plt.plot(obsX,obsY,'sk')
    plt.plot(searhx,searhy,'sr')

    ax.set_xlim([-1,cols])
    ax.set_ylim([-1,rows])

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    plt.pause(0.05)


opt_pathIndex = path[goalIndex]
optpathsub = []

for i in range(len(opt_pathIndex)):
    optpathsub.append(map.ind2sub([rows,cols],opt_pathIndex[i]))

optx = []
opty = []

for i in range(0,len(optpathsub),1):
    field[optpathsub[i][0]][optpathsub[i][1]] =6
    optxy = map.sub2xy([rows,cols],optpathsub[i][0],optpathsub[i][1])
    optx.append(optxy[0])
    opty.append(optxy[1])

field[startSub[0],startSub[1]] =4
field[goalSub[0],goalSub[1]] = 5

plt.figure()
plt.plot(startXY[0],startXY[1],'r+')
plt.plot(goalXY[0],goalXY[1],'b+')
plt.plot(obsX,obsY,'sk')
plt.plot(searhx,searhy,'sr')
plt.plot(optx,opty,'b')

ax.set_xlim([-1,cols])
ax.set_ylim([-1,rows])

ax.set_xticks(np.arange(cols))
ax.set_yticks(np.arange(rows))

plt.show()






