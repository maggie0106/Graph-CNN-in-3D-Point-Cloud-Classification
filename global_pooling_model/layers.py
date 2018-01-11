import tensorflow as tf
def weightVariables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.05)
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    chebyshevWeights = dict()
    for i in range(chebyshevOrder):
        initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=0.05)
        chebyshevWeights['w_' + str(i)] = tf.Variable(initial)
    return chebyshevWeights


def gcnLayer(inputPC, scaledLaplacian, pointNumber, inputFeatureN, outputFeatureN, chebyshev_order):
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    chebyPoly = []
    cheby_K_Minus_1 = tf.matmul(scaledLaplacian, inputPC)
    cheby_K_Minus_2 = inputPC
    chebyPoly.append(cheby_K_Minus_2)
    chebyPoly.append(cheby_K_Minus_1)
    for i in range(2, chebyshev_order):
        chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
	#chebyK = tf.matmul(scaledLaplacian, cheby_K_Minus_1)
        chebyPoly.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
        #cheby_K_Minus_2, cheby_K_Minus_1 = cheby_K_Minus_1, chebyK
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
    #l2_max_pooling_pre = tf.reshape(gcnOutput, [-1, 1024, featureNumber, 1])
    #max_pooling_output_1=tf.nn.max_pool(l2_max_pooling_pre,ksize=[1,1024,1,1],strides=[1,1,1,1],padding='VALID')
    #max_pooling_output_1=tf.reshape(max_pooling_output_1,[-1,featureNumber])
    #mean, var = tf.nn.moments(gcnOutput, axes=[1])
    #poolingOutput = tf.concat([max_pooling_output_1, var], axis=1)

    mean, var = tf.nn.moments(gcnOutput, axes=[1])
    max_f = tf.reduce_max(gcnOutput, axis=[1])
    poolingOutput = tf.concat([max_f, var], axis=1)
    #return max_f
    return poolingOutput

#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    return outputFC
