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
        chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
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
    max_f = tf.reduce_max(gcnOutput, axis=[1])
    poolingOutput = tf.concat([max_f, var], axis=1)
    return poolingOutput


def fullyConnected(features, inputFeatureN, outputFeatureN):
    # Description: fully connected layer without relu activation
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    return outputFC
