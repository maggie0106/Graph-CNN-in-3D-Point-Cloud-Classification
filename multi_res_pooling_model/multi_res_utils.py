#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:26:47 2017

@author: yingxuezhang
"""
import numpy as np
from scipy.spatial import cKDTree

def cluster_index(inputTrain, inputTrainCentroid, clusterNumber,nearestNeighbor):
    cluster_index_dict = {}
    for j in range(len(inputTrain)):
        chunk_input = inputTrain[j]
        chunk_centroid = inputTrainCentroid[j]
        
        chunk_index = np.zeros([len(chunk_input), clusterNumber*nearestNeighbor])
        chunk_index = chunk_index.astype(int)
        
        for i in range(len(chunk_input)):
            object_xyz = chunk_input[i]
            object_centroid = chunk_centroid[i]
            tree = cKDTree(object_xyz)
            dd, ii = tree.query(object_centroid, k = nearestNeighbor)
            index_select = ii.flatten()
            chunk_index[i] = index_select
        cluster_index_dict.update({j:chunk_index})
    print "Finish calculate the cluster index"
    return cluster_index_dict


def centroid_point_extract(inputData, clusterNumber):
    centroidPoint = {}
    for i in range(len(inputData)):
        inputCoor = inputData[i]
        centroidPoint.update({i:inputCoor[:,0:clusterNumber,:]})
    return centroidPoint