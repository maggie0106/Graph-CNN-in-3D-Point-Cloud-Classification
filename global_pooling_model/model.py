import tensorflow as tf
import numpy as np
from layers import gcnLayer, globalPooling, fullyConnected
from utils import get_mini_batch, add_noise, weights_calculation, uniform_weight
from sklearn.metrics import confusion_matrix

# ===========================Hyper parameter=====================
def model_architecture(para):
    inputPC = tf.placeholder(tf.float32, [None, para.pointNumber, 3])
    inputGraph = tf.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber])
    outputLabel = tf.placeholder(tf.float32, [None, para.outputClassN])

    scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])

    weights = tf.placeholder(tf.float32, [None])
    lr = tf.placeholder(tf.float32)
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)

    # gcn layer 1
    gcn_1 = gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=3,
                     outputFeatureN=para.gcn_1_filter_n,
                     chebyshev_order=para.chebyshev_1_Order)
    gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
    gcn_1_pooling = globalPooling(gcn_1_output, featureNumber=para.gcn_1_filter_n)
    print("The output of the first gcn layer is {}".format(gcn_1_pooling))
    print gcn_1_pooling

    # gcn_layer_2
    
    gcn_2 = gcnLayer(gcn_1_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_1_filter_n,
                     outputFeatureN=para.gcn_2_filter_n,
                     chebyshev_order=para.chebyshev_2_Order)
    gcn_2_output = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)
    gcn_2_pooling = globalPooling(gcn_2_output, featureNumber=para.gcn_2_filter_n)
    print("The output of the second gcn layer is {}".format(gcn_2_pooling))
    
    #gcn_layer_3
    '''
    gcn_3 = gcnLayer(gcn_2_output, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.gcn_2_filter_n,
                     outputFeatureN=para.gcn_3_filter_n,
                     chebyshev_order=para.chebyshev_2_Order)
    gcn_3_output = tf.nn.dropout(gcn_3, keep_prob=keep_prob_1)
    gcn_3_pooling = globalPooling(gcn_3_output, featureNumber=para.gcn_3_filter_n)
    print("The output of the second gcn layer is {}".format(gcn_2_pooling))
    '''

    # concatenate global features
    #globalFeatures = gcn_3_pooling
    globalFeatures = tf.concat([gcn_1_pooling, gcn_2_pooling], axis=1)
    globalFeatures = tf.nn.dropout(globalFeatures, keep_prob=keep_prob_2)
    print("The global feature is {}".format(globalFeatures))
    #globalFeatureN = para.gcn_2_filter_n*2
    globalFeatureN = (para.gcn_1_filter_n + para.gcn_2_filter_n)*2 

    # fully connected layer 1
    fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
    fc_layer_1 = tf.nn.relu(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
    print("The output of the first fc layer is {}".format(fc_layer_1))

    # fully connected layer 2
    fc_layer_2 = fullyConnected(fc_layer_1, inputFeatureN=para.fc_1_n, outputFeatureN=para.outputClassN)
    print("The output of the second fc layer is {}".format(fc_layer_2))

    # =================================Define loss===========================
    predictSoftMax = tf.nn.softmax(fc_layer_2)
    predictLabels = tf.argmax(predictSoftMax, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_2, labels=outputLabel)
    loss = tf.multiply(loss, weights)
    loss = tf.reduce_mean(loss)

    vars = tf.trainable_variables()
    loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 8e-6  # best: 8 #last: 10
    loss_total = loss + loss_reg

    correct_prediction = tf.equal(predictLabels, tf.argmax(outputLabel, axis=1))
    acc = tf.cast(correct_prediction, tf.float32)
    acc = tf.reduce_mean(acc)

    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print('Total parameters number is {}'.format(total_parameters))
    
    trainOperaion = {'train': train, 'loss_total':loss_total,'loss': loss, 'acc': acc, 'loss_reg': loss_reg, 'inputPC': inputPC,
                     'inputGraph': inputGraph, 'outputLabel': outputLabel, 'weights': weights,
                     'predictLabels': predictLabels,
                     'keep_prob_1': keep_prob_1, 'keep_prob_2': keep_prob_2, 'lr': lr}

    return trainOperaion


# (input data) input inputCoor, input Graph, input Label dictionary -------3
# (model training)train, loss, acc,loss_l2 -------------3
# (hyper parameter)batchSize, keep_prob, keep_prob_1,lr:learning_rate
# Return average loss, acc, reg


from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize


def trainOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion, weight_dict, learningRate):
    dataChunkLoss = []
    dataChunkAcc = []
    dataChunkRegLoss = []
    for i in range(len(inputCoor)):
        xTrain_1, graphTrain_1, labelTrain_1 = inputCoor[i], inputGraph[i], inputLabel[i]
        graphTrain_1 = graphTrain_1.tocsr()
        labelBinarize = label_binarize(labelTrain_1, classes=[j for j in range(para.outputClassN)])
        xTrain, graphTrain, labelTrain = shuffle(xTrain_1, graphTrain_1, labelBinarize)
        # labelBinarize = label_binarize(labelTrain, classes=[j for j in range(40)])

        batch_loss = []
        batch_acc = []
        batch_reg = []
        batchSize = para.batchSize
        for batchID in range(len(labelBinarize) / para.batchSize):
            start = batchID * batchSize
            end = start + batchSize
            batchCoor, batchGraph, batchLabel = get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
            batchGraph = batchGraph.todense()


            batchCoor = add_noise(batchCoor, sigma=0.008, clip=0.02)
            if para.weighting_scheme == 'uniform':
                batchWeight = uniform_weight(batchLabel)
            elif para.weighting_scheme == 'weighted':
                batchWeight = weights_calculation(batchLabel, weight_dict)
            else:
                print 'please enter the valid weighting scheme'
	        
	    #print batchWeight

            feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                         trainOperaion['outputLabel']: batchLabel, trainOperaion['lr']: learningRate,
                         trainOperaion['weights']: batchWeight,
                         trainOperaion['keep_prob_1']: para.keep_prob_1, trainOperaion['keep_prob_2']: para.keep_prob_2}

            opt, loss_train, acc_train, loss_reg_train = sess.run(
                [trainOperaion['train'], trainOperaion['loss_total'], trainOperaion['acc'], trainOperaion['loss_reg']],
                feed_dict=feed_dict)

            #print('The loss loss_reg and acc for this batch is {},{} and {}'.format(loss_train, loss_reg_train, acc_train))
            batch_loss.append(loss_train)
            batch_acc.append(acc_train)
            batch_reg.append(loss_reg_train)

        dataChunkLoss.append(np.mean(batch_loss))
        dataChunkAcc.append(np.mean(batch_acc))
        dataChunkRegLoss.append(np.mean(batch_reg))


    train_average_loss = np.mean(dataChunkLoss)
    train_average_acc = np.mean(dataChunkAcc)
    loss_reg_average = np.mean(dataChunkRegLoss)
    return train_average_loss, train_average_acc, loss_reg_average


def evaluateOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion):
    test_loss = []
    test_acc = []
    test_predict = []
    for i in range(len(inputCoor)):
        xTest, graphTest, labelTest = inputCoor[i], inputGraph[i], inputLabel[i]
        graphTest = graphTest.tocsr()
        labelBinarize = label_binarize(labelTest, classes=[i for i in range(para.outputClassN)])
        test_batch_size = para.testBatchSize
        for testBatchID in range(len(labelTest) / test_batch_size):
            start = testBatchID * test_batch_size
            end = start + test_batch_size
            batchCoor, batchGraph, batchLabel = get_mini_batch(xTest, graphTest, labelBinarize, start, end)
            batchWeight = uniform_weight(batchLabel)
            batchGraph = batchGraph.todense()

            feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                         trainOperaion['outputLabel']: batchLabel, trainOperaion['weights']: batchWeight,
                         trainOperaion['keep_prob_1']: 1.0, trainOperaion['keep_prob_2']: 1.0}

            predict, loss_test, acc_test = sess.run(
                [trainOperaion['predictLabels'], trainOperaion['loss'], trainOperaion['acc']], feed_dict=feed_dict)
            test_loss.append(loss_test)
            test_acc.append(acc_test)
            test_predict.append(predict)

    test_average_loss = np.mean(test_loss)
    test_average_acc = np.mean(test_acc)

    return test_average_loss, test_average_acc, test_predict
