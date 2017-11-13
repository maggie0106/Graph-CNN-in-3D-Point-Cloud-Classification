import tensorflow as tf
from layers import gcnLayer, graph_cluster_maxpooling, fullyConnected
from utils import get_mini_batch, add_noise, weights_calculation, uniform_weight, farthest_sampling, middle_graph_generation
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
import numpy as np


def model_architecture(para):
    # Description: build model architecture (build data flow graphs)
    # Input: global parameter instance
    # Return: Placeholder Dictionary

    inputPC = tf.placeholder(tf.float32, [None, para.pointNumber, 3])
    inputGraph = tf.placeholder(tf.float32, [None, para.pointNumber * para.pointNumber])
    l2Graph = tf.placeholder(tf.float32, [None, para.clusterNumberL1 * para.clusterNumberL1])
    outputLabel = tf.placeholder(tf.float32, [None, para.outputClassN])
    batch_size = tf.placeholder(tf.int32)

    batch_index_l1 = tf.placeholder(tf.int32, [None, para.clusterNumberL1 * para.nearestNeighborL1])
    # batch_index_l2 = tf.placeholder(tf.int32, [None, para.clusterNumberL2 * para.nearestNeighborL2])

    scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])
    l2_scaledLaplacian = tf.reshape(l2Graph, [-1, para.clusterNumberL1, para.clusterNumberL1])

    weights = tf.placeholder(tf.float32, [None])
    lr = tf.placeholder(tf.float32)
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)

    # gcn layer 1
    gcn_1 = gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=3,
                     outputFeatureN=para.gcn_1_filter_n,
                     chebyshev_order=para.chebyshev_1_Order)
    gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
    # multi-resolution pooling layer 1
    gcn_1_pooling = graph_cluster_maxpooling(batch_index_l1, gcn_1_output, batch_size=batch_size,
                                             M=para.clusterNumberL1, k=para.nearestNeighborL1, n=para.gcn_1_filter_n)
    print gcn_1_pooling

    gcn_2 = gcnLayer(gcn_1_pooling, l2_scaledLaplacian, pointNumber=para.clusterNumberL1,
                     inputFeatureN=para.gcn_1_filter_n,
                     outputFeatureN=para.gcn_2_filter_n, chebyshev_order=para.chebyshev_1_Order)

    gcn_2_pooling = tf.nn.dropout(gcn_2, keep_prob=keep_prob_1)

    print gcn_2_pooling
    globalFeatures = tf.reduce_max(gcn_2_pooling, axis=1)
    print globalFeatures

    globalFeatures = tf.nn.dropout(globalFeatures, keep_prob=keep_prob_2)
    print("The global feature is {}".format(globalFeatures))

    globalFeatureN = para.gcn_2_filter_n

    # fully connected layer 1
    fc_layer_1 = fullyConnected(globalFeatures, inputFeatureN=globalFeatureN, outputFeatureN=para.fc_1_n)
    fc_layer_1 = tf.nn.relu(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob = keep_prob_2)
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
    loss = loss + loss_reg

    correct_prediction = tf.equal(predictLabels, tf.argmax(outputLabel, axis=1))
    acc = tf.cast(correct_prediction, tf.float32)
    acc = tf.reduce_mean(acc)

    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print('Total parameters number is {}'.format(total_parameters))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    trainOperaion = {'train': train, 'loss': loss, 'acc': acc, 'loss_reg': loss_reg, 'inputPC': inputPC,
                     'inputGraph': inputGraph, 'l2Graph': l2Graph, 'outputLabel': outputLabel, 'weights': weights,
                     'predictLabels': predictLabels, 'batch_index_l1': batch_index_l1,
                     'keep_prob_1': keep_prob_1, 'keep_prob_2': keep_prob_2, 'lr': lr, 'batch_size': batch_size}
    return trainOperaion, sess


def trainOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion,
                  weight_dict, learningRate):
    # Description: training one epoch (two options to train the model, using weighted gradient descent or normal gradient descent)
    # Input: (1)inputCoor: input coordinates (B, N, 3) (2) inputGraph: input graph (B, N*N) (3) inputLabel: labels (B, 1)
    #        (4) para: global Parameters  (5) sess: Session (6) trainOperaion: placeholder dictionary
    #        (7) weight_dict: weighting scheme used of weighted gradient descnet (8)learningRate: learning rate for current epoch
    # Return: average loss, acc, regularization loss for training set


    dataChunkLoss = []
    dataChunkAcc = []
    dataChunkRegLoss = []
    for i in range(len(inputLabel)):
        xTrain_1, graphTrain_1, labelTrain_1 = inputCoor[i], inputGraph[i], inputLabel[i]

        graphTrain_1 = graphTrain_1.tocsr()
        labelBinarize = label_binarize(labelTrain_1, classes=[j for j in range(40)])
        xTrain, graphTrain, labelTrain = shuffle(xTrain_1, graphTrain_1, labelBinarize)

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
            # batchWeight = weights_calculation(batchLabel, weight_dict)
            batchWeight = uniform_weight(batchLabel)
            # select the centroid points by farthest sampling and get the index of each centroid points' n nearest neighbors
            batchIndexL1, centroid_coordinates = farthest_sampling(batchCoor, M=para.clusterNumberL1,
                                                                   k=para.nearestNeighborL1, batch_size=batchSize,
                                                                   nodes_n=para.pointNumber)
            # setting up the graph structure for the second gcn layer(constructed by centroid points)
            batchMiddleGraph = middle_graph_generation(centroid_coordinates, batch_size = batchSize, M = para.clusterNumberL1)

            feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                         trainOperaion['outputLabel']: batchLabel, trainOperaion['lr']: learningRate,
                         trainOperaion['weights']: batchWeight,
                         trainOperaion['keep_prob_1']: para.keep_prob_1, trainOperaion['keep_prob_2']: para.keep_prob_2,
                         trainOperaion['batch_index_l1']: batchIndexL1,
                         trainOperaion['l2Graph']: batchMiddleGraph, trainOperaion['batch_size']: para.batchSize}

            opt, loss_train, acc_train, loss_reg_train = sess.run(
                [trainOperaion['train'], trainOperaion['loss'], trainOperaion['acc'], trainOperaion['loss_reg']],
                feed_dict=feed_dict)

            batch_loss.append(loss_train)
            batch_acc.append(acc_train)
            batch_reg.append(loss_reg_train)

            #print "The loss, L2 loss and acc for this batch is {}, {} and {}".format(loss_train, loss_reg_train, acc_train)

        dataChunkLoss.append(np.mean(batch_loss))
        dataChunkAcc.append(np.mean(batch_acc))
        dataChunkRegLoss.append(np.mean(batch_reg))

    train_average_loss = np.mean(dataChunkLoss)
    train_average_acc = np.mean(dataChunkAcc)
    loss_reg_average = np.mean(dataChunkRegLoss)
    return train_average_loss, train_average_acc, loss_reg_average


def evaluateOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion):
    # Description: Performance on the test set data
    # Input: (1)inputCoor: input coordinates (B, N, 3) (2) inputGraph: input graph (B, N*N) (3) inputLabel: labels (B, 1)
    #        (4) para: global Parameters  (5) sess: Session (6) trainOperaion: placeholder dictionary
    # Return: average loss, acc, regularization loss for test set
    test_loss = []
    test_acc = []
    test_predict = []
    for i in range(len(inputCoor)):
        xTest, graphTest, labelTest = inputCoor[i], inputGraph[i], inputLabel[i]
        graphTest = graphTest.tocsr()
        labelBinarize = label_binarize(labelTest, classes=[j for j in range(40)])
        test_batch_size = para.testBatchSize
        for testBatchID in range(len(labelTest) / test_batch_size):
            start = testBatchID * test_batch_size
            end = start + test_batch_size
            batchCoor, batchGraph, batchLabel = get_mini_batch(xTest, graphTest, labelBinarize, start, end)
            batchWeight = uniform_weight(batchLabel)
            batchGraph = batchGraph.todense()
            # select the centroid points by farthest sampling and get the index of each centroid points n nearest neighbors
            batchIndexL1, centroid_coordinates = farthest_sampling(batchCoor, M=para.clusterNumberL1,
                                                                   k=para.nearestNeighborL1, batch_size=test_batch_size,
                                                                   nodes_n=para.pointNumber)

            batchMiddleGraph = middle_graph_generation(centroid_coordinates, batch_size = test_batch_size, M = para.clusterNumberL1)


            feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                         trainOperaion['outputLabel']: batchLabel, trainOperaion['weights']: batchWeight,
                         trainOperaion['keep_prob_1']: 1.0, trainOperaion['keep_prob_2']: 1.0,
                         trainOperaion['batch_index_l1']: batchIndexL1,
                         trainOperaion['l2Graph']: batchMiddleGraph, trainOperaion['batch_size']: test_batch_size
                         }

            predict, loss_test, acc_test = sess.run(
                [trainOperaion['predictLabels'], trainOperaion['loss'], trainOperaion['acc']], feed_dict=feed_dict)
            test_loss.append(loss_test)
            test_acc.append(acc_test)
            test_predict.append(predict)

    test_average_loss = np.mean(test_loss)
    test_average_acc = np.mean(test_acc)

    return test_average_loss, test_average_acc, test_predict
