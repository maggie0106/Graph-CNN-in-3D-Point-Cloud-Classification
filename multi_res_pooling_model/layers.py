import tensorflow as tf
def weightVariables(shape, name):
    # Description: define weight matrix
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.05)
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    # Description: define weight matrix in graph convolutional layer
    chebyshevWeights = dict()
    for i in range(chebyshevOrder):
        initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=0.05)
        chebyshevWeights['w_' + str(i)] = tf.Variable(initial)
    return chebyshevWeights


def gcnLayer(inputPC, scaledLaplacian, pointNumber, inputFeatureN, outputFeatureN, chebyshev_order):
    # Description: graph convolutional layer with Relu activation
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyPoly = []
    cheby_K_Minus_1 = tf.matmul(scaledLaplacian, inputPC)
    cheby_K_Minus_2 = inputPC

    chebyPoly.append(cheby_K_Minus_2)
    chebyPoly.append(cheby_K_Minus_1)
    for i in range(2, chebyshev_order):
        chebyK = 2*tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
        chebyPoly.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK

    chebyOutput = []
    for i in range(chebyshev_order):
        weights = chebyshevCoeff['w_' + str(i)]
        chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, inputFeatureN])
        output = tf.matmul(chebyPolyReshape, weights)
        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        chebyOutput.append(output)
    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    gcnOutput = tf.nn.relu(gcnOutput)
    return gcnOutput


def globalPooling(gcnOutput, featureNumber):
    # Description: pooling layer with global statistical pooling containing max and variance of each feature map
    mean, var = tf.nn.moments(gcnOutput, axes=[1])
    max_pooling = tf.reduce_max(gcnOutput, axis=[1])
    poolingOutput = tf.concat([max_pooling, var], axis=1)
    return poolingOutput


def graph_cluster_maxpooling(batch_index, batch_feature_maps, batch_size,M, k, n):
    # Description: max pooling on each of the cluster
    # input: (1)index function: B*M*k (batch_index)
    #       (2)feature maps: B*N*n1 (batch_feature_maps)
    #       (3) batch_size
    #       (4) cluster size M
    #       (5) nn size k
    #       (6) n feature maps dimension
    # output: (1)B*M*n1 (after max-pooling on each of the cluster)
    index_reshape = tf.reshape(batch_index, [M*k*batch_size, 1])
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1))
    batch_idx_tile = tf.tile(batch_idx, (1, M * k))
    batch_idx_tile_reshape = tf.reshape(batch_idx_tile, [M*k*batch_size, 1])
    new_index = tf.concat([batch_idx_tile_reshape, index_reshape], axis=1)
    group_features = tf.gather_nd(batch_feature_maps, new_index)

    group_features_reshape = tf.reshape(group_features, [batch_size, M, k, n])
    max_features = tf.reduce_max(group_features_reshape, axis=2)
    return max_features


#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    # Description: fully connected layer without relu activation
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    return outputFC
