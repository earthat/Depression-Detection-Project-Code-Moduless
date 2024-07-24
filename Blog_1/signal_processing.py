import numpy as np
import scipy.signal
import h5py
import pandas as pd
from functools import partial

class SignalProcessing:

    @staticmethod
    def bandpass_filter(edges, sample_rate, poles, data):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data

    @staticmethod
    def notch_filter(frequency_toRemove, quality_factor, sample_frequency, data):
        b, a = scipy.signal.iirnotch(frequency_toRemove, quality_factor, sample_frequency)
        return b, a

    @staticmethod
    def resample(data, sample_frequency, downsample_frequency):
        n = data.shape[1]
        num = (n//sample_frequency)*downsample_frequency
        return scipy.signal.resample(data, num, axis=1)

    @staticmethod
    def data_augmentation(signal, sample_freq, window_size, stride_size):
        N = signal.shape[1]
        f = window_size*sample_freq
        s = stride_size*sample_freq
        n = (N-f+s)//s
        lst = [signal[:, i*s:i*s+f] for i in range(n)]
        tensor = np.stack(lst, axis=0)
        return tensor

    @staticmethod
    def fn_tensor(signal, n_slices):
        signal_slices = np.hsplit(signal, n_slices)
        tensor = np.stack(signal_slices, axis=0)
        return tensor

    @staticmethod
    def tensor_generator(signal, sample_freq, time_slice_size):
        n = signal.shape[1]
        assert n%sample_freq == 0
        t = n//sample_freq
        assert t%time_slice_size == 0
        n_time_slices = t//time_slice_size
        return SignalProcessing.fn_tensor(signal, n_time_slices)

    @staticmethod
    def signal_processing(np_array, sample_freq, band_freq_range, downsample_freq, slice_size, sub_slice_size):
        bandpass_filter = partial(SignalProcessing.bandpass_filter, band_freq_range, sample_freq, 5)
        signal_processed1 = np.apply_along_axis(bandpass_filter, 1, np_array)
        signal_processed2 = SignalProcessing.resample(signal_processed1, sample_freq, downsample_freq)
        window_size = slice_size
        stride_size = window_size // 3
        signal_augmented = SignalProcessing.data_augmentation(signal_processed2, downsample_freq, window_size, stride_size)
        signal_tensors = [SignalProcessing.tensor_generator(signal_slice, downsample_freq, sub_slice_size) for signal_slice in signal_augmented]
        tensor = np.stack(signal_tensors, axis=0)
        return tensor

    @staticmethod
    def AdjMatrix(df_electrodes_coordinates):
        np_coordinates = np.array(df_electrodes_coordinates)
        _, b = np_coordinates.shape
        lst = []
        for i in range(b):
            f = np_coordinates[:,i]
            d = f.reshape(-1,1)-f.reshape(1,-1)
            d_square = d**(2)
            lst.append(d_square)
        tensor = np.stack(lst, axis=0)
        tensor_dist_sq = tensor.sum(axis=0)
        tensor_dist = tensor_dist_sq**(1/2)
        max_val, min_val = tensor_dist.min(), tensor_dist.max()
        tensor_dist_norm = (tensor_dist-min_val)/(max_val-min_val)
        tensor_dist_norm = tensor_dist_norm if np.linalg.det(tensor_dist_norm) else np.array([])

        return tensor_dist_norm

    @staticmethod
    def to_bdi_label(bdi):
        bdi_label = None
        if bdi < 7:
            bdi_label = 0  # Not depression
        elif bdi >= 17:
            bdi_label = 1  # Depression
        return bdi_label

    @staticmethod
    def to_h5(file_path, subject, signal_array, label, frequency):
        with h5py.File(file_path, "w") as hf:
                hf.create_dataset("subject", data=subject)
                hf.create_dataset("resampled_signal", data=signal_array)
                hf.create_dataset("resample_freq", data=frequency)
                hf.create_dataset("label", data=label)
        return

    @staticmethod
    def signal_processing_iterator(data_lst, sample_freq, band_freq_range, downsample_freq, slice_size, sub_slice_size):
        correct_data_lst = []
        error_data_lst = []
        for subject_id, signal_np, df_coordinates, bdi in data_lst:
            signal_processed = SignalProcessing.signal_processing(signal_np, sample_freq, band_freq_range, downsample_freq, slice_size, sub_slice_size)
            Adj_dist_matrix = SignalProcessing.AdjMatrix(df_coordinates)
            bdi_label = SignalProcessing.to_bdi_label(bdi)
            if bdi_label in [0, 1] and Adj_dist_matrix.size:
                correct_data_lst.append((subject_id, signal_processed, Adj_dist_matrix, bdi, bdi_label))
            else:
                error_data_lst.append((subject_id, signal_processed, Adj_dist_matrix, bdi, bdi_label))

        return correct_data_lst, error_data_lst
