
class Parameters():
    def __init__(self):
	self.pointNumber = 1024
	self.neighborNumber = 50
        self.outputClassN = 40
        self.pointNumber = 1024
        self.gcn_1_filter_n = 1200 # filter number of the first gcn layer
        self.gcn_2_filter_n = 1000 # filter number of the second gcn layer
        self.fc_1_n = 600          # fully connected layer dimension
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.9
        self.keep_prob_2 = 0.55
        self.batchSize = 28
        self.testBatchSize = 1
        self.learningRate = 12e-4
        self.maxEpoch = 300
        self.samplingType = 'farthest_sampling'
        self.modelDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/model/'
        self.logDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/log/'
        self.fileName = '1113_nn_50'
        self.weight_scaler = 50 # weight scaler for weighted gradient descent


