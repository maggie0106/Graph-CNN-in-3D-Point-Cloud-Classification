from read_data import load_data, prepareData
import tensorflow as tf
from model import model_architecture, trainOneEpoch, evaluateOneEpoch
import numpy as np
from Parameters import Parameters
from utils import weight_dict_fc
from sklearn.metrics import confusion_matrix
import pickle
import time

start_time = time.time()
samplingType = 'farthest_sampling'
# ===============================Hyper parameters========================
para = Parameters()
print 'Dataset {}'.format(para.dataset)
print 'The point number and the nearest neighbor number is {} and {}'.format(para.pointNumber, para.neighborNumber)
print 'The first and second layer filter number is {} and {}'.format(para.gcn_1_filter_n, para.gcn_2_filter_n)
print 'The fc neuron number is {} and the output number is {}'.format(para.fc_1_n, para.outputClassN)
print 'The Chebyshev polynomial order for each layer are {} and {}'.format(para.chebyshev_1_Order, para.chebyshev_2_Order)
print 'The weighting scheme is {} and the weighting scaler is {}'.format(para.weighting_scheme, para.weight_scaler)


#=======================================================================

pointNumber = para.pointNumber
neighborNumber = para.neighborNumber

with tf.Graph().as_default():
# ===============================Build model=============================
    trainOperaion = model_architecture(para)
    #init = tf.global_variables_initializer()
    #sess = tf.Session()
    #sess.run(init)
    # ================================Load data===============================
    if para.dataset == 'ModelNet40':
        inputTrain, trainLabel, inputTest, testLabel = load_data(pointNumber, samplingType)
    elif para.dataset == 'ModelNet10':
        ModelNet10_dir = '/raid60/yingxue.zhang2/ICASSP_code/data/'
        with open(ModelNet10_dir+'input_data','rb') as handle:
            a = pickle.load(handle)
        inputTrain, trainLabel, inputTest, testLabel = a 
    else:
        print "Please enter a valid dataset"
    scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, neighborNumber, pointNumber)

    # ===============================Train model ================================
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    learningRate = para.learningRate

    modelDir = para.modelDir
    save_model_path = modelDir + "model_" + para.fileName
    weight_dict = weight_dict_fc(trainLabel, para)
    testLabelWhole = []
    for i in range(len(testLabel)):
        labels = testLabel[i]
        [testLabelWhole.append(j) for j in labels]
    testLabelWhole = np.asarray(testLabelWhole)

    test_acc_record = []
    test_mean_acc_record = []
    
    for epoch in range(para.max_epoch):
        print('===========================epoch {}===================='.format(epoch))
        if (epoch % 20 == 0):
            learningRate = learningRate /2 #1.7
        learningRate = np.max([learningRate, 10e-6])
        print(learningRate)
        train_average_loss, train_average_acc, loss_reg_average = trainOneEpoch(inputTrain, scaledLaplacianTrain,
                                                                                trainLabel,
                                                                                para, sess, trainOperaion, weight_dict,
                                                                                learningRate)

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

        # save log
        log_Dir = para.logDir
        fileName = para.fileName
        with open(log_Dir + 'confusion_mat_' + fileName, 'wb') as handle:
            pickle.dump(confusion_mat, handle)
        print('the average acc among 40 class is:{}'.format(mean_class_acc))
        print(
        '===========average loss and acc for this epoch is {} and {}======='.format(test_average_loss, test_average_acc))
        test_acc_record.append(test_average_acc)
        test_mean_acc_record.append(mean_class_acc)

        with open(log_Dir + 'overall_acc_record_' + fileName, 'wb') as handle:
            pickle.dump(test_acc_record, handle)
        with open(log_Dir + 'mean_class_acc_record_' + fileName, 'wb') as handle:
            pickle.dump(test_mean_acc_record, handle)
end_time = time.time()
run_time = (end_time - start_time)/3600
print 'The running time for this trail is {} hours'.format(run_time)	
