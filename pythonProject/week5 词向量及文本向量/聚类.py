import numpy as np
import random
import sys

'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer: # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)


    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        # ndarray为维矩阵，如100 * 8，遍历所有的点
        for item in self.ndarray:
            distance_min = sys.maxsize # 9223372036854775807
            index = -1
            # 对于某个固定的点，假设分为10个簇，计算这个点离哪个质心是最近的
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()] # 划分到离得最近的簇里
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    #计算距离总和
    def __sumdis(self, result):
        sum = 0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum += self.__distance(result[i][j], self.points[i])
    
        return sum
    
    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)
        
    # 欧式距离
    def __distance(self, p1, p2):
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i] , 2)
        return pow(tmp, 0.5) 
    
    # 选取cluster_num个点，作为质心
    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception('族数设置有误')
        #取点的下标，随机的数量是cluster_num个
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)
    

x = np.random.rand(100, 8) # 100 * 8
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)