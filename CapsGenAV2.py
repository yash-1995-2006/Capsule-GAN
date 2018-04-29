import tensorflow as tf
import numpy as np
from operator import mul

batchSize=64


def squash(capsule, epsilon=1e-9):
    '''
    :param vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1]
    :param epsilon: delta to prevent zero division
    :return A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    norm = tf.norm(capsule, axis=2)
    factor = tf.expand_dims(tf.divide(norm, tf.add(1.0, tf.add(tf.square(norm),epsilon))), 2)
    return tf.multiply(capsule, factor)


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def capsgen(x, initCapsuleSize = 32, batchSize = 64, isTrain = True):
    '''
    :param x: batch of randomly generated numbers (batch size x capsule length)
    :param initCapsuleSize: initial capsule size
    :return: image of (batch size x 32 x 32)
    '''
    #normalize the vector to make size of capsule 1
    x = tf.nn.l2_normalize(x, dim=1)
    x = tf.expand_dims(tf.expand_dims(x,axis=2), axis=1, name='expanded_x')    #[batch size x number of capsules x capsules length x 1]
    #capsule16, W16, C16, B16= capslayer(x, layerNo=1, capsuleLength=16, numberOfCapsules=10, stddev=0.05)                        #[batch size x number of capsules x capsules length x 1]
    capsule8, W8, C8, B8 = capslayer(x, layerNo=1, capsuleLength=8, numberOfCapsules=1152, stddev=0.1)                #[batch size x number of capsules x capsules length x 1]
    convImageSize = 6
    reshapedCaps = tf.reshape(capsule8, shape=[batchSize,convImageSize, convImageSize,-1], name="reshapedCaps")      #[batch size x x_dim x y_dim x number of filters]
    conv1 = tf.layers.conv2d_transpose(reshapedCaps,filters=128,kernel_size=[4,4], padding="valid", name='Conv1')
    relu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

    conv2 = tf.layers.conv2d_transpose(relu1, 64, [4, 4], strides=(1, 1), padding='valid', name='Conv2')
    relu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

    conv3 = tf.layers.conv2d_transpose(relu2, 32, [5, 5], strides=(1, 1), padding='valid', name='Conv3')
    relu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

    conv4 = tf.layers.conv2d_transpose(relu3, 1, [4, 4], strides=(2, 2), padding='same', name='Conv4')
    relu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

    tanh = tf.nn.tanh(relu4,name='tanh')
    return tanh, W8, C8, B8



def generateNoisyVector(vector, outputCapsuleNumber, uniformNoise=False, uniformAvg = False, uniformMin = False, uniformNoiseRatio=1.0, stddev=1.0):
    '''
    :param vector: input vector to replicate and add noise to for Routing [batch size x number of input capsules x capsule length x 1]
    :param outputCapsuleNumber: Number of Capsules in the required layer
    :param uniformNoise: generate noise belong
    :return: [batch size x number of input caps x number of output caps x capsule size x 1]
    '''
    print("in gNV ", vector)
    #increase vector dimensions
    inputCapsuleLength = tf.shape(vector)[-2]
    vector = tf.tile(tf.expand_dims(vector, axis=2),[1,1,outputCapsuleNumber,1,1]) #[[batch size x number of input capsules x number of output capsules x capsule length x 1]
    if uniformNoise == False:
        #add random noise to half the elements of the tensor
        noise = tf.random_normal(shape=tf.shape(vector),stddev=stddev)
        vector = tf.add(vector, noise)
    else:
        if uniformAvg == True:
            metric = tf.reduce_mean(vector,axis=3,keepdims=True, name='metric')
        elif uniformMin == True:
            metric = tf.reduce_min(vector,axis=3,keepdims=True, name='metric')
        noise = tf.random_uniform(shape=tf.shape(vector),minval=-1*uniformNoiseRatio,maxval=uniformNoiseRatio, name='initial_noise')
        expandMetric = tf.tile(metric, (1,1,1,inputCapsuleLength,1), name='expanded_metric')
        noise = tf.multiply(expandMetric, noise, name='multiplied_noise')
        vector = tf.add(vector,noise)
    return vector







def modifiedDynamicRouting(inputCaps,outputCapsuleNumber, layerNo, iter=3, stddev=1.0):
    '''
    :param inputCaps:input capsule values [batch size x number of capsules x capsule length x 1]
    :param outputCapsuleNumber: number of  capsules in output
    :param iter: number of routing iterations
    :param stddev: standard deviation for controlling then noise added
    :param layerNo: Dynamic Routing for given layer number
    :return: [batch size x inputCapsuleNumber x outputCapsuleNumber x capsule length x 1]
    '''
    with tf.variable_scope("mDR" + str(layerNo)):
        print("in mDR",layerNo," ",inputCaps)
        inputShape = inputCaps.get_shape().as_list()
        numberOfInputCaps = inputShape[1]
        B = tf.Variable(name='B'+str(layerNo), trainable=False, initial_value=tf.random_normal([numberOfInputCaps, outputCapsuleNumber-1, 1, 1], dtype=tf.float32))
        expandedInput = generateNoisyVector(inputCaps, outputCapsuleNumber-1, uniformNoise=True,uniformMin=True,uniformNoiseRatio=0.5)
        for i in range(iter):
            with tf.variable_scope('iter_' + str(i)):
                C = tf.nn.softmax(B, dim=1)
                agreedValues = tf.multiply(expandedInput, C)                
                sumExpandedCaps = tf.reduce_sum(agreedValues, axis=2)
                extraCaps = tf.expand_dims(tf.subtract(inputCaps,sumExpandedCaps), axis=2)
                if i == iter - 1:
                    agreedValues = tf.concat([agreedValues,extraCaps],axis=2)
                if i < iter - 1:                    
                    inputCapsExpanded = tf.expand_dims(inputCaps, dim=2)                    
                    A = tf.reduce_sum(tf.reduce_sum(tf.multiply(agreedValues, inputCapsExpanded), axis=3, keepdims=True),)
                    B += A
        return agreedValues, C, B





def capslayer(x, capsuleLength, numberOfCapsules, layerNo, stddev, routing = 'Modified Dynamic Routing'):
    '''
    :param x: input activation vectors [batch size x number of capsules x capsules length x 1]
    :param capsuleLength: Length of required capsules
    :param numberOfCapsules: Number of capsules in the layer
    :param: layerNo: Capsule layer number
    :param routing: routing method
    :return: capsule output vectors
    '''
    inputShape = x.get_shape().as_list()
    #batchSize = inputShape[0]
    inputCapsNum = inputShape[1]
    inputCapsLen = inputShape[-2:]
    with tf.variable_scope("Capsule"+str(layerNo)):
        if routing == 'Modified Dynamic Routing':
            routedValues, C, B = modifiedDynamicRouting(x,numberOfCapsules,layerNo=layerNo, stddev=stddev)    #[batch size x inputCapsuleNumber x outputCapsuleNumber x input capsule length x 1]
        W = tf.Variable(name='W'+str(layerNo), trainable=True, initial_value=tf.random_normal([inputCapsNum, numberOfCapsules, capsuleLength, inputCapsLen[0]]))
        Wx = tf.tile(tf.expand_dims(W,axis=0),[batchSize,1,1,1,1])      #[batch size x inputCapsNumber x numberOfCapsules, capsuleLength, input Capsule Length]
        weightedVectors = tf.matmul(Wx, routedValues)                    #[batch size x inputCapsuleNumber x outputCapsuleNumber x output capsule length x 1]
        reducedWeightedVectors = tf.reduce_sum(weightedVectors, axis=1) #[batch size x outputCapsuleNumber x output capsule length x 1]
        capsule = squash(reducedWeightedVectors)
    return capsule, W, C, B

