import numpy as np
import pandas as pd
import os
import scipy.io
import scipy.signal
import mne

channelsList = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'CP3', 'CP4', 'CP5', 'CP6']

class SignalExtraction():

    @staticmethod
    def fn_signal_extraction_openneuro(raw_signal_obj, channels_subset=None):
        dict_x, dict_y = {}, {}
        for channel in raw_signal_obj.ch_names:
            signal_channel = raw_signal_obj[channel]
            y, x = signal_channel
            y, x = y.flatten(), x.flatten()
            dict_y[channel] = y
            dict_x[channel] = x

        signal_df = pd.DataFrame(dict_x)
        signal_df = signal_df[channels_subset] if channels_subset else signal_df
        signal_np = np.array(signal_df).transpose()
        return signal_np

    @staticmethod
    def fn_signal_extraction_predict_d006(raw_signal_obj, channels_subset=None):
        raw_signal = raw_signal_obj['EEG'][0][0]
        channels = [i[0][0] for i in raw_signal[21].reshape(-1)]
        signals = raw_signal[15]
        control_treatment = raw_signal[0][0]
        label = raw_signal[-2][0]
        assert signals.shape[0] == len(channels)
        signals = signals.transpose()
        signal_df = pd.DataFrame(signals, columns=channels)
        signal_df = signal_df[channels_subset] if channels_subset else signal_df
        signal_np = np.array(signal_df).transpose()
        return signal_np, control_treatment, label

    @staticmethod
    def read_mat_iterator(path, num=None):
        lst = []
        i = 0
        skip_subjects = [
            'sub-065', 'sub-080', 'sub-084', 'sub-052', 'sub-053', 'sub-055', 'sub-023', 'sub-032', 'sub-033',
            'sub-003', 'sub-015', 'sub-012', 'sub-017', 'sub-019', 'sub-021', 'sub-022', 'sub-025', 'sub-026',
            'sub-027', 'sub-058', 'sub-059', 'sub-061', 'sub-063', 'sub-096', 'sub-106', 'sub-028', 'sub-002',
            'sub-020', 'sub-004', 'sub-006', 'sub-097', 'sub-103', 'sub-101', 'sub-008', 'sub-024', 'sub-030',
            'sub-031', 'sub-035', 'sub-034', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-041', 'sub-042',
            'sub-043', 'sub-045', 'sub-044'
        ]

        for root, _, files in os.walk(path):
            if any(sub in root for sub in skip_subjects):
                continue  # Skip processing this folder

            for name in files:
                i += 1
                mat_file = os.path.join(root, name)
                mat = scipy.io.loadmat(mat_file)
                signal_np, control_treatment, label = SignalExtraction.fn_signal_extraction_predict_d006(mat, channelsList)
                lst.append((signal_np, control_treatment, label))
                if num and i == num:
                    return lst
        return lst

    @staticmethod
    def get_participants_data_openneuro(file):
        df_participants = pd.read_csv(file, delimiter="\t")
        return df_participants

    @staticmethod
    def get_electrodes_coordinates(file, channelsList):
        df_coordinates = pd.read_csv(file, delimiter="\t")
        df1 = df_coordinates[df_coordinates.name.map(lambda x: x in channelsList)]
        df2 = df1.set_index('name')
        df3 = df2.reindex(channelsList)
        return df3

    @staticmethod
    def fn_label_extraction_openneuro(subject, df):
        BDI_obj = df[df.participant_id == subject].BDI
        return BDI_obj.values[0]

    @staticmethod
    def read_mne_iterator(path, num=None):
        lst = []
        i = 0
        skip_subjects = [
            'sub-065', 'sub-080', 'sub-084', 'sub-052', 'sub-053', 'sub-055', 'sub-023', 'sub-032', 'sub-033',
            'sub-003', 'sub-015', 'sub-012', 'sub-017', 'sub-019', 'sub-021', 'sub-022', 'sub-025', 'sub-026',
            'sub-027', 'sub-058', 'sub-059', 'sub-061', 'sub-063', 'sub-096', 'sub-106', 'sub-028', 'sub-002',
            'sub-020', 'sub-004', 'sub-006', 'sub-097', 'sub-103', 'sub-101', 'sub-008', 'sub-024', 'sub-030',
            'sub-031', 'sub-035', 'sub-034', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-041', 'sub-042',
            'sub-043', 'sub-045', 'sub-044'
        ]

        df = SignalExtraction.get_participants_data_openneuro(path + 'participants.tsv')

        def fn_read_EEG(root, run, channelsList):
            set_file = run['set']
            coordiantes_file = run['electrodes']
            name_split = set_file.split('_')
            subject_id = name_split[0] + '_' + name_split[2]
            bdi = SignalExtraction.fn_label_extraction_openneuro(name_split[0], df)
            mne_file = os.path.join(root, set_file)
            mne_obj = mne.io.read_raw_eeglab(mne_file)
            signal_np = SignalExtraction.fn_signal_extraction_openneuro(mne_obj, channelsList)
            coordiantesFile = os.path.join(root, coordiantes_file)
            df_coordinates = SignalExtraction.get_electrodes_coordinates(coordiantesFile, channelsList)
            return subject_id, signal_np, df_coordinates, bdi

        for root, _, files in os.walk(path):
            if any(sub in root for sub in skip_subjects):
                continue  # Skip processing this folder

            if root.endswith('eeg'):
                run2 = {}
                for name in files:
                    if '_run-02_' in name:
                        if '_eeg.set' in name:
                            run2['set'] = name
                        if '_electrodes.tsv' in name:
                            run2['electrodes'] = name

                print('<<<<<<<Processing {}>>>>>>>'.format(root))
                if run2:
                    try:
                        subject_id, signal_np, df_coordinates, bdi = fn_read_EEG(root, run2, channelsList)
                        lst.append((subject_id, signal_np, df_coordinates, bdi))
                    except Exception as e:
                        print(f'<<<<<<<Error in {run2}: {e}>>>>>>>')
                        continue
                i += 1
                if num and i == num:
                    return lst
        return lst
