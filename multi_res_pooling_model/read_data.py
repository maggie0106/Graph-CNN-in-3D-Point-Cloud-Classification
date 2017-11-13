#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:59 2017

@author: yingxuezhang
"""
import os
import utils
import scipy
from utils import adjacency, scaled_laplacian
import numpy as np
from scipy.spatial import cKDTree
import pickle
def farthestSampling(file_names, NUM_POINT):
    #Description: load point cloud data into dictionary format {key: batch index, value: batch data or labels}
    #               containing each batch of the data (data prepared by farthest sampling)
    #input: (1)file name (2) point number
    #return: (1) input data dictionary (2) input data label
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label) 
        current_label= np.int_(current_label)
        inputData.update({index : current_data})
        inputLabel.update({index : current_label})
    return inputData, inputLabel

def uniformSampling(file_names, NUM_POINT):
    #Description: load point cloud data into dictionary format {key: batch index, value: batch data or labels}
    #                       containing each batch of the data (data prepared by uniform sampling)

    #input: (1)file name (2) point number
    #return: (1) input data dictionary (2) input data label
    file_indexs = np.arange(0, len(file_names))
    inputData = dict()
    inputLabel = dict()
    for index in range (len(file_indexs)):
        current_data, current_label = utils.loadDataFile(file_names[file_indexs[index]])
        current_label = np.squeeze(current_label) 
        current_label= np.int_(current_label)
        output = np.zeros((len(current_data), NUM_POINT, 3))
        for i,object_xyz in enumerate (current_data):
            samples_index=np.random.choice(2048, NUM_POINT, replace=False)
            output[i] = object_xyz[samples_index]
        inputData.update({index : output})
        inputLabel.update({index : current_label})
    return inputData, inputLabel

# ModelNet40 official train/test split
def load_data(NUM_POINT, sampleType):
    #Description: load training and testing data and its corresponding label
    #input: (1)NUM_POINT: point number (2)sampleType: data preparation method
    #return: (1) training set data  (2) training set label
    #        (3) testing set data   (4) testing set label

    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))


    BASE_DIR = '/raid60/yingxue.zhang2/ICASSP_code/'    
    TRAIN_FILES = utils.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    TEST_FILES = utils.getDataFiles(\
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    
    #np.random.shuffle(train_file_idxs)
    if sampleType == 'farthest_sampling':
        inputTrainFarthest, inputTrainLabel = farthestSampling(TRAIN_FILES, NUM_POINT)
        inputTestFathest, inputTestLabel = farthestSampling(TEST_FILES, NUM_POINT)
        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel
    
    elif sampleType == 'uniform_sampling':
        inputTrainFarthest, inputTrainLabel = uniformSampling(TRAIN_FILES, NUM_POINT)
        inputTestFathest, inputTestLabel = uniformSampling(TEST_FILES, NUM_POINT)

        return inputTrainFarthest, inputTrainLabel, inputTestFathest, inputTestLabel



#generate graph structure and store in the system
def prepareGraph(inputData, neighborNumber, pointNumber, dataType):
    #Description: generate graph structure and store in the system for reuse
    #input: (1)inputData: input point coordinates (2)neighborNumber: neighbor number when constructing nearest neighbor graph
    #       (3)pointNumber: point number of each object (4)dataType: training set or testing set
    #return: scaled Laplacian matrix of each object

    scaledLaplacianDict = dict()
    #baseDir = os.path.dirname(os.path.abspath(__file__))
    baseDir ='/raid60/yingxue.zhang2/ICASSP_code'  
    #baseDir= os.path.abspath(os.path.dirname(os.getcwd()))
    fileDir =  baseDir+ '/graph/' + dataType+'_pn_'+str(pointNumber)+'_nn_'+str(neighborNumber)
    if (not os.path.isdir(fileDir)):
        print "calculating the graph data"
        os.makedirs(fileDir)
        for batchIndex in range(len(inputData)):
            batchInput = inputData[batchIndex]
            for i in range(len(batchInput)):
                print i
                pcCoordinates = batchInput[i]
                tree = cKDTree(pcCoordinates)
                dd, ii = tree.query(pcCoordinates, k = neighborNumber)
                A = adjacency(dd, ii)
                scaledLaplacian = scaled_laplacian(A)
                flattenLaplacian = scaledLaplacian.tolil().reshape((1, pointNumber*pointNumber))
                if i ==0:
                    batchFlattenLaplacian = flattenLaplacian
                else:
                    batchFlattenLaplacian = scipy.sparse.vstack([batchFlattenLaplacian, flattenLaplacian])
            scaledLaplacianDict.update({batchIndex: batchFlattenLaplacian})
            with open(fileDir+'/batchGraph_'+str(batchIndex), 'wb') as handle:
                pickle.dump(batchFlattenLaplacian, handle)
            print "Saving the graph data batch"+str(batchIndex)
        
    else:
        print("Loading the graph data from "+dataType+'Data')
        scaledLaplacianDict = loadGraph(inputData, neighborNumber, pointNumber, fileDir)
    return scaledLaplacianDict


def loadGraph(inputData, neighborNumber, pointNumber, fileDir):
    #Description: load graph from RAM if it already exists
    scaledLaplacianDict = dict()
    for batchIndex in range(len(inputData)):
        batchDataDir = fileDir+'/batchGraph_'+str(batchIndex)
        with open(batchDataDir, 'rb') as handle:
            batchGraph = pickle.load(handle)
        scaledLaplacianDict.update({batchIndex: batchGraph })
        print "Finish loading batch_"+str(batchIndex)
    return scaledLaplacianDict
        
                        
def prepareData(inputTrain, inputTest, neighborNumber, pointNumber):
    #Description: prepare input graph data
    # input: (1)inputTrain: training data (2)inputTest: testing data
    # (3) neighborNumber: neighbor number when constructing nearest neighbor graph (4) pointNumber: point number of object
    scaledLaplacianTrain = prepareGraph(inputTrain, neighborNumber, pointNumber, 'train',)
    scaledLaplacianTest = prepareGraph(inputTest, neighborNumber, pointNumber, 'test')
    return scaledLaplacianTrain, scaledLaplacianTest
    
