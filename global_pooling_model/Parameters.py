
class Parameters():
    def __init__(self):
	self.pointNumber = 1024
	self.neighborNumber = 50
        self.outputClassN = 40
        self.pointNumber = 1024
        self.gcn_1_filter_n = 1200
        self.gcn_2_filter_n = 1000
        self.fc_1_n = 600
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.9
        self.keep_prob_2 = 0.55
        self.batchSize = 28
        self.testBatchSize = 1
        self.learningRate = 12e-4
        self.modelDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/model/'
        self.logDir = '/raid60/yingxue.zhang2/ICASSP_code/global_pooling/log/'
        self.fileName = '1113_nn_50'
        self.weight_scaler = 50


