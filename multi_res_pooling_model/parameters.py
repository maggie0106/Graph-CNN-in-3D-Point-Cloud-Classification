
class parameters():
    def __init__(self):
        #address
        self.modelDir = '/raid60/yingxue.zhang2/ICASSP_code/multi_res_random/model/'
        self.logDir = '/raid60/yingxue.zhang2/ICASSP_code/multi_res_random/log/'
        self.fileName = '1111_multi_res'

        #fix parameters
        self.neighborNumber = 50
        self.pointNumber = 1024
        self.outputClassN = 40
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 3
        self.keep_prob_1 = 0.8
        self.keep_prob_2 = 0.55
        self.batchSize = 28
        self.testBatchSize = 1
        self.learningRate = 12e-4
        self.weight_scaler = 40

        self.gcn_1_filter_n = 1000
        self.gcn_2_filter_n = 1000
        self.fc_1_n = 300 #600

        #multi res parameters
        self.clusterNumberL1 = 35
        self.nearestNeighborL1 = 56

        self.clusterNumberL2 = 4
        self.nearestNeighborL2 = 10
        
        


