#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:59 2017

@author: yingxuezhang
"""
from read_data import load_data, prepareData
import tensorflow as tf
from model import model_architecture, trainOneEpoch, evaluateOneEpoch
import numpy as np
from Parameters import Parameters
from utils import weight_dict_fc
from sklearn.metrics import confusion_matrix
import pickle

# ===============================Load parameters========================
para = Parameters()
pointNumber = para.pointNumber
neighborNumber = para.neighborNumber

with tf.Graph().as_default():
    # ===============================Build model=============================
    trainOperaion = model_architecture(para)
    # ================================Load data==============================
    inputTrain, trainLabel, inputTest, testLabel = load_data(pointNumber, para.samplingType)
    scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, neighborNumber, pointNumber)
    # ===============================Train model ================================
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    modelDir = para.modelDir
    save_model_path = modelDir + "model_" + para.fileName
    weight_dict = weight_dict_fc(trainLabel, para)

    #ground truth for the test set
    testLabelWhole = []
    for i in range(len(testLabel)):
        labels = testLabel[i]
        [testLabelWhole.append(j) for j in labels]
    testLabelWhole = np.asarray(testLabelWhole)

    test_acc_record = []
    test_mean_acc_record = []
    learningRate = para.learningRate
    for epoch in range(para.maxEpoch):
        print('===========================epoch {}===================='.format(epoch))

        #decay learning rate every 20 epoch
        if (epoch % 20 == 0):
            learningRate = learningRate / 1.7
        learningRate = np.max([learningRate, 1e-5])
        print(learningRate)
        train_average_loss, train_average_acc, loss_reg_average = trainOneEpoch(inputTrain, scaledLaplacianTrain,
                                                                                trainLabel,
                                                                                para, sess, trainOperaion, weight_dict,
                                                                                learningRate)
        #save model after every epoch
        save = saver.save(sess, save_model_path)
        print('=============average loss, l2 loss, acc  for this epoch is {} {} and {}======'.format(train_average_loss,
                                                                                                     loss_reg_average,
                                                                                                     train_average_acc))

        test_average_loss, test_average_acc, test_predict = evaluateOneEpoch(inputTest, scaledLaplacianTest, testLabel,
                                                                             para, sess, trainOperaion)

        # calculate mean class accuracy
        test_predict = np.asarray(test_predict)
        test_predict = test_predict.flatten()
        confusion_mat = confusion_matrix(testLabelWhole[0:len(test_predict)], test_predict)
        normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1)
        class_acc = np.diag(normalized_confusion)
        mean_class_acc = np.mean(class_acc)

        # save log(mean instance acc and mean class acc)
        log_Dir = para.logDir
        fileName = para.fileName
        with open(log_Dir + 'confusion_mat_' + fileName, 'wb') as handle:
            pickle.dump(confusion_mat, handle)
        print('the average acc among 40 class is:{}'.format(mean_class_acc))
        print('===========average loss and acc for this epoch is {} and {}======='.format(test_average_loss,
                                                                                          test_average_acc))
        test_acc_record.append(test_average_acc)
        test_mean_acc_record.append(mean_class_acc)

        with open(log_Dir + 'overall_acc_record_' + fileName, 'wb') as handle:
            pickle.dump(test_acc_record, handle)
        with open(log_Dir + 'mean_class_acc_record_' + fileName, 'wb') as handle:
            pickle.dump(test_mean_acc_record, handle)

