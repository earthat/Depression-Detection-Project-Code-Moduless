import numpy as np

class TrainingData:
    
    @staticmethod
    def signal_to_trainingData(signal_tuple):
        ls = []
        subject_id, signal_processed, Adj_dist_matrix, bdi, bdi_label = signal_tuple
        for signal_slice in signal_processed:
            ls.append((signal_slice, Adj_dist_matrix, bdi_label))
        return ls

    @staticmethod
    def trainingData_iterator(signal_tuples):
        ls = []
        for signal_tuple in signal_tuples:
            ls = ls + TrainingData.signal_to_trainingData(signal_tuple)
        return ls
