import numpy as np
from common import time
from collections import namedtuple
from sklearn import preprocessing
# from GraphLearning.MIT_dataset_utils import load_edf_data
from numpy import fft
from scipy.signal import hilbert
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
from GraphLearning.utils import load_EU_features, load_side_adj, load_EU_settings, sec2win, win2sec, \
     class_relabel, rolling_window, EU_online_fileNum_RollingWindow, fileNum2winNum
# from GraphLearning.utils import inpython_online_wrapper, NN_classifier, RandomForest_classifier



class LoadCore(object): 
    def __init__(self, preszr_sec, postszr_sec, band_nums, idx_szr=None, idx_nonszr=None, idx_preszr=None):
        self.preszr_sec = preszr_sec
        self.postszr_sec = postszr_sec
        self.band_nums = band_nums
        self.idx_szr = idx_szr
        self.idx_nonszr = idx_nonszr
        self.idx_preszr = idx_preszr
        
        

class Task(object): 
    def __init__(self, task_core):
        self.task_core = task_core



def X_flat_complex(arr):
    arr_flat = arr.reshape(arr.shape[0], -1)
    return np.concatenate((np.real(arr_flat), np.imag(arr_flat)), axis=1)
     
def PreProcessing(X, y, multiClass_ratio, clip_sizes, data_load_core):
    if(multiClass_ratio is not None and multiClass_ratio != -1):
        szr_args = np.argwhere(y!=0)[:,0]
        nonszr_args = np.argwhere(y==0)[:,0]
        num_sel = np.min((np.size(szr_args)*multiClass_ratio, y.size))
        nonszr_args = np.random.choice(nonszr_args, num_sel, replace=False)
        sel_idx = np.sort(np.hstack((szr_args, nonszr_args)))
        X, y = X[sel_idx,...], y[sel_idx] # np.concatenate((X[y!=0,...], X[sel_idx,...]), axis=0), np.concatenate((y[y!=0], y[sel_idx]), axis=0)
    if(not np.any(np.array(data_load_core.dimArray)==1) and data_load_core.band_nums is not None):
        aggregate_conv_sizes = np.concatenate(([0], np.cumsum(data_load_core.conv_sizes)), axis=0).astype(np.int)
        print('aggregate_conv_sizes: ',  aggregate_conv_sizes)
        band_idx = np.hstack([np.arange(start=aggregate_conv_sizes[band_num],stop=aggregate_conv_sizes[band_num+1],step=1) \
                                    for band_num in data_load_core.band_nums])
        print('band_idx: ',  band_idx)
        X = X[:, :, band_idx, :]
    conv_sizes = data_load_core.conv_sizes[data_load_core.band_nums] if data_load_core.band_nums is not None else data_load_core.conv_sizes
    return X, y , clip_sizes, conv_sizes, X.shape[2:]
    



class online_load():
    def __init__(self, data_load_core, task_core):
        self.batch_num = data_load_core.settings_TrainNumFiles
        self.data_load_core = data_load_core # data_load_core.settings_numFiles
        self.matlab_engin = matlab.engine.start_matlab()
        self.task_core = task_core
        
    def current_x(self):
        return self.dataX
    
    def current_y(self):
        return self.dataY
    
    def next(self):
        self.batch_num += 1
        self.dataX, self.dataY, sel_win_nums, conv_sizes, clip_sizes = \
                    inpython_online_wrapper(self.matlab_engin, self.data_load_core, [self.batch_num], 'total', self.task_core.data_dir)
        
        return 
    def end(self):
        return self.batch_num > self.data_load_core.settings_TrainNumFiles + self.data_load_core.settings_TestNumFiles



# def train_test_splitting(y, split_ratio, clip_sizes=None):
#     random.seed(0)
#     def rand_sel(arr, indx, split_ratio): # from array, choose half of 1's and half of 0's randomly
#         szr_indx = np.squeeze(np.argwhere(arr!=0))
#         szr_samples = random.sample(list(indx[szr_indx]), int(split_ratio*szr_indx.size))
#         nonszr_indx = np.squeeze(np.argwhere(arr==0))
#         nonszr_samples = random.sample(list(indx[nonszr_indx]), int(split_ratio*nonszr_indx.size))
#         return np.concatenate((szr_samples,nonszr_samples))
#         
#     if(clip_sizes is None): # later: shuffle and stuff
#         return rand_sel(y, np.arange(y.size), split_ratio)
#     else: # later
#         # from each clip, choose half of 1's and half of 0's randomly
#         train_idx = np.concatenate([rand_sel(y[np.arange(clip_sizes[i,0],clip_sizes[i,1])], np.arange(clip_sizes[i,0],clip_sizes[i,1]), split_ratio) 
#                                                     for i in range(clip_sizes.shape[0])])
#         test_idx = np.array(list(set(np.arange(y.size))-set(train_idx)))
#         return train_idx, test_idx
    
    
def h5py2complex(f, keyy):
    try:
        temp = f[keyy].value
    except:
        temp = f[keyy]
        print('f[key]: ', temp.shape)
        print(temp.view(np.double).shape)
    try:
        arr = temp.view(np.double).reshape(np.concatenate((temp.shape , [2])))
        out = np.nan_to_num(arr[...,0] + 1j*arr[...,1])
    except:
        out = temp.view(np.double)
    return out


class WrapperDataLoad():
    
    def __init__(self, matFile):
        conv_sizes = np.array(matFile['conv_sizes'])
        self.conv_sizes = np.reshape(conv_sizes, (conv_sizes.size,))
        
        soz_ch_ids = np.array(matFile['soz_ch_ids'])
        self.soz_ch_ids = np.reshape(soz_ch_ids, (soz_ch_ids.size,))
        
        self.pre_ictal_num_win = int(np.array(matFile['n_pre_szr']))
        
        self.window_size_sec = 2.5 # np.float32(np.array(matFile['window_size_sec']))
        self.stride_sec = 1.5 # np.float32(np.array(matFile['stride_sec']))
        
        self.X_train = h5py2complex(matFile, 'X_train')
#         print('preee xtrain shape: ', self.X_train.shape)
        if(self.X_train.ndim == 3):
            self.X_train = self.X_train[np.newaxis, ...]
#             self.X_train = np.swapaxes(self.X_train, 0, 1)
        
        y = np.nan_to_num(np.array(matFile['y_train']))
        self.y_train = np.reshape(y, (y.size, )) # np.any(y<0) # np.unique(y)
        
        
        sel_win_nums = np.array(matFile['sel_win_nums_train'])
        self.sel_win_nums_train = np.reshape(sel_win_nums, (sel_win_nums.size,))
        self.clip_sizes_train = np.array(matFile['clip_sizes_train'])
        
        self.X_test = h5py2complex(matFile, 'X_test')
#         print('preee xtest shape: ', self.X_test.shape)
        if(self.X_test.ndim == 3):
            self.X_test = self.X_test[np.newaxis, ...]
        
        
        y = np.nan_to_num(np.array(matFile['y_test']))
        self.y_test = np.reshape(y, (y.size,))
        
        
        sel_win_nums = np.array(matFile['sel_win_nums_test'])
        self.sel_win_nums_test = np.reshape(sel_win_nums, (sel_win_nums.size,))
        self.clip_sizes_test = np.array(matFile['clip_sizes_test'])
        
        if(self.y_train.size != self.X_train.shape[0]):
            self.X_train = np.swapaxes(np.swapaxes(self.X_train, 0, 3), 1, 2)
        if(self.y_test.size != self.X_test.shape[0]):
            self.X_test = np.swapaxes(np.swapaxes(self.X_test, 0, 3), 1, 2)
        
        
class WrapperDataToy():
    def __init__(self):
        return
    
Structural_Side_Info = namedtuple('Structural_Side_Info', ['adj_means','adj_vars'])    
class data_load(Task):
    
    
    def EU_matlab_run(self, load_Core):
        start_time = time.get_seconds()
        
#         load_Core.matlab_engin.loading_EU_main(nargout=0)
#         X = load_Core.matlab_engin.workspace['X']
#         y = load_Core.matlab_engin.workspace['y']
        
        matFile = load_EU_features(self.task_core.data_dir , self.task_core.target, load_Core)
        
        data_load_core = WrapperDataLoad(matFile)
        data_load_core.band_nums = load_Core.band_nums
        data_load_core.target = self.task_core.target
#         data_load_core.y_train = class_relabel(data_load_core.y_train, data_load_core.clip_sizes_train, data_load_core, \
#                                                  preszr_sec=load_Core.preszr_sec, postszr_sec=load_Core.postszr_sec)
#         if(data_load_core.y_test is not None and data_load_core.y_test.size >1 ):
#             data_load_core.y_test = class_relabel(data_load_core.y_test, data_load_core.clip_sizes_test, data_load_core, \
#                                                     preszr_sec=load_Core.preszr_sec, postszr_sec=load_Core.postszr_sec)
            
        data_load_core.structural_inf = load_side_adj(self.task_core)
        
        print('    X training', data_load_core.X_train.shape, 'y training', data_load_core.y_train.shape)
        print('    X testing', data_load_core.X_test.shape, 'y testing', data_load_core.y_test.shape)
        
        data_load_core.settings_TrainNumFiles, data_load_core.settings_TestNumFiles = load_EU_settings(self.task_core.settings_dir , self.task_core.target, load_Core)
        dim_Array = data_load_core.X_train.shape[2:]
        data_load_core.dimArray = dim_Array
        print('data_load_core.dimArray', data_load_core.dimArray)
        print('    time elapsed: ', time.get_seconds()-start_time)
        return  data_load_core.X_train.shape[1],  dim_Array, data_load_core
    
    
#     def MITCHB_run(self, load_Core):
#         start_time = time.get_seconds()
#         out_data = load_edf_data(self.task_core.data_dir , self.task_core.target, load_Core)
#         num_clips = []
#         if(load_Core.concat):
#             X = None
#             y = None
#         else:
#             X = []
#             y = []
#         
#         for data, file_name, seizure_start_time_offsets, seizure_lengths in out_data:
#             inner_x, inner_y, num_nodes, dim, conv_sizes = windowing_data(data, seizure_start_time_offsets, seizure_lengths, load_Core)
# #             print('inner_x: ', np.array(inner_x).shape)
# #             num_clips.append(np.array(inner_x).shape[0])
#             if(load_Core.concat):
#                 if(X is None):
#                     X = np.array(inner_x)
#                     y = np.array(inner_y)
#                 else:
#                     X = np.concatenate((X,inner_x), axis=0)
#                     y = np.concatenate((y,inner_y), axis=0)
#             else:
#                 X.append(inner_x)
#                 y.append(inner_y)
#     
#         X = np.array(X)
#         y = np.array(y)
#         print('    X', X.shape, 'y', y.shape)
#         print('    time elapsed: ', time.get_seconds()-start_time)
#         return X, y, num_nodes, dim, conv_sizes  #, num_clips
    
    def toy_data_gen(self,load_Core):
        num_nodes = 5
        dimArray = [num_nodes*3,1]
        n_train_samp = 10
        n_test_samp = 100
        data_load_core = WrapperDataToy()
        data_load_core.target = self.task_core.target
        data_load_core.X_train = np.random.rand(n_train_samp, num_nodes, dimArray[0], dimArray[1])
        data_load_core.y_train = np.random.randint(0, high=2, size=n_train_samp)
        data_load_core.y_train[0] = 1
        data_load_core.X_test = np.random.rand(n_test_samp, num_nodes, dimArray[0], dimArray[1])
        data_load_core.y_test = np.random.randint(low=0, high=2, size=n_test_samp)
        data_load_core.dimArray = dimArray
        data_load_core.conv_sizes = dimArray[0]
        data_load_core.clip_sizes_train = None
        data_load_core.clip_sizes_test = None
        data_load_core.band_nums = None
        data_load_core.sel_win_nums_test = None
        adj = np.random.randint(0, high=2, size=(num_nodes,num_nodes))
        adj = ((adj+adj.T)/2).astype(int)
        data_load_core.structural_inf = Structural_Side_Info(adj_means=[adj],adj_vars=[np.zeros((num_nodes,num_nodes))])
        return num_nodes, dimArray, data_load_core
    
    
    def run(self, load_Core):
        print ('Loading ECoG Data ...') 
        if('ECoG' in self.task_core.signal_mode) :
            return self.EU_matlab_run(load_Core)
#         if(load_Core.dataSet =='EU'):
#             return self.EU_matlab_run(load_Core)
#         else:
#             return self.MITCHB_run(load_Core)
        
        
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)
    return X_train, X_cv


def train_classifier(classifier, data, normalize=False):
    X_train = data.X_train
    y_train = data.y_train
    X_cv = data.X_cv
    y_cv = data.y_cv
    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)
    print( "Training ...")
    print( 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv))
    start = time.get_seconds()
    classifier.fit(X_train, y_train)
    print ("Scoring...")
    S, E = score_classifier_auc(classifier, X_cv, y_cv, data.y_classes)
    score = 0.5 * (S + E)

    elapsedSecs = time.get_seconds() - start
    print ("t=%ds score=%f" % (int(elapsedSecs), score))
    
    return {
        'classifier': classifier,
        'score': score,
        'S_auc': S,
        'E_auc': E
    }

class DownSampler():
    def __init__(self, load_core):
        self.load_core = load_core
        
    def apply(self, X):
        T = X.shape[-1]
        ratio = self.load_core.down_sampl_ratio
#         output = np.zeros((S, ratio))
        window_len = int(T/ratio)
#         if(window_len<2):
#             return X
#         for i in range(ratio):
#             output[:,i] = np.mean(X[:,i * window_len:(i+1) * window_len], axis=1)    
        output = X[...,0:window_len:T]
        return output
    
    
class NoChange():  
    def __init__(self, load_core):
        self.load_core = load_core
    def apply(self, X):
        return X   
        
class Normalize():
    def __init__(self, load_core):
        self.load_core = load_core
        
    def apply(self, X):
        y= (X-np.mean(X, axis=1)[:, np.newaxis])/np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis])
        if(self.load_core.down_sampl_ratio is not None):
            y = DownSampler(self.load_core).apply(y)
        return y
    
class FFT():
    def __init__(self):
        return
        
    def apply(self, X_raw, load_core):
        win_length = int(np.ceil(load_core.welchs_win_len * load_core.sampling_freq))
        X = rolling_window(X_raw, win_length, int(np.ceil(load_core.welchs_stride * load_core.sampling_freq)))
        X = np.swapaxes(np.swapaxes(X, 0, 1), 1, 2)
        f_signal = rfft(X) 
        W = fftfreq(f_signal.shape[-1], d=1/load_core.sampling_freq)

#         f_signal = np.swapaxes(f_signal, 2, 3)
#         if(load_core.down_sampl_ratio is not None):
#             f_signal = DownSampler(load_core).apply(f_signal)
        conv_sizes = []
        all_sizess = np.zeros_like(W)
        for i in np.arange(len(load_core.freq_bands)):
            if(i>0):
                lowcut = load_core.freq_bands[i-1]
            else:
                lowcut = load_core.initial_freq_band
            highcut = load_core.freq_bands[i]
            sizess = np.where(W<highcut, np.ones_like(W), np.zeros_like(W))
            sizess = np.where(W<lowcut, np.zeros_like(W), sizess)
            all_sizess += sizess
            conv_sizes.append(int(np.sum(sizess)*f_signal.shape[-2]))
        in_FFT_W = f_signal[...,np.squeeze(np.argwhere(all_sizess==1))]
        FFT_W = np.reshape(in_FFT_W, (in_FFT_W.shape[0], in_FFT_W.shape[1], in_FFT_W.shape[2]*in_FFT_W.shape[3]))
#         W = np.tile(W,np.hstack((f_signal.shape[:-1],1)))
#         FFT_W = None        
#         for i in np.arange(len(load_core.freq_bands)):
#             if(i>0):
#                 lowcut = load_core.freq_bands[i-1]
#             else:
#                 lowcut = load_core.initial_freq_band
#             highcut = load_core.freq_bands[i]
# #             butter_bandpass_filter(data, lowcut, highcut, fs)
#             cut_f_signal = f_signal.copy()
# #             cut_f_signal = np.where(W<highcut, cut_f_signal,0 ) 
# #             cut_f_signal = np.where(W>=lowcut, cut_f_signal,0 ) 
#             cut_f_signal[ W >= highcut] = 0 # np.abs(W)
#             cut_f_signal[ W < lowcut] = 0
#             cut_f_signal = np.reshape(cut_f_signal, np.hstack((cut_f_signal.shape[:-2],np.multiply(cut_f_signal.shape[-2],cut_f_signal.shape[-1]))))# check again if correct ?????????????
#             if(load_core.down_sampl_ratio is not None):
#                 cut_f_signal = DownSampler(load_core).apply(cut_f_signal)
#             if(FFT_W is None):
#                 FFT_W = cut_f_signal
#             else:
#                 FFT_W = np.concatenate((FFT_W,cut_f_signal), axis=-1)
#             conv_sizes.append(cut_f_signal.shape[-1])
        return FFT_W, np.array(conv_sizes)
    
      
def data_convert(x, load_core):
    conversion_names = load_core.data_conversions #.split('+')
    y= None
    conv_sizes = None
    for i in range(len(conversion_names)):
#         exec("y.append(%s(load_core).apply(x))" % (conversion_names[i]) )
        conv, new_conv_sizes = conversion_names[i].apply(x, load_core)
        y = conv if y is None else np.concatenate((y,conv),axis=-1)
        conv_sizes = new_conv_sizes if conv_sizes is None else np.hstack((conv_sizes, new_conv_sizes))
#     yy=None
#     for i in range(len(conversion_names)):
#         yy=y[i] if yy is None else np.concatenate((y[i],yy),axis=1)
    
    return np.array(y), conv_sizes

class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, start, end=None):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d%s" % (self.start, '-%d' % self.end if self.end is not None else '')

    def apply(self, data, meta=None):
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]

def to_np_array(X):
    if isinstance(X[0], np.ndarray):
        # return np.vstack(X)
        out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
        for i, x in enumerate(X):
            out[i] = x
        return out

    return np.array(X)


             
def windowing_data(input_data, seizure_start_time_offsets, seizure_lengths, load_Core):
    S,T = input_data.shape
#     feature_extraction.graphL_core.num_nodes = S
    sampling_freq = load_Core.sampling_freq
    if(load_Core.num_windows is not None):
        win_len_sec = T/(sampling_freq * load_Core.num_windows)
        stride_sec = win_len_sec
    else:
        win_len_sec = 2.5
        stride_sec = 1.5  
        
    seizure_start_time_offsets *= sampling_freq
    seizure_lengths *= sampling_freq
    win_len = int(np.ceil(win_len_sec * sampling_freq))
    stride = int(np.ceil(stride_sec * sampling_freq))
    X = rolling_window(input_data, win_len, stride) # np.swapaxes(, 0, 1) # np.lib.stride_tricks.as_strided(input_data, strides = st, shape = (input_data.shape[1] - w + 1, w))[0::o]
    if(load_Core.down_sampl_ratio is not None):
        X = DownSampler(load_Core).apply(X)
    X, conv_sizes = data_convert(X, load_Core) # X
    intervals_with_stride = np.arange(0, T - win_len, stride) # rolling_window(np.arange(T)[np.newaxis,:], win_len, stride) # 
    num_windows = len(intervals_with_stride)
    print('    win_len_sec: %f , num_windows: %d' % (win_len_sec, num_windows))
    y = np.zeros((num_windows,))
    flag_ictal = False
    y_detection = 1 if load_Core.detection_flag else 0
    def state_gen(y, win_ind):
        if(len(load_Core.state_win_lengths)<1):
            return y
        state_counter = 2 if load_Core.detection_flag else 1
        num_winds = list(np.ceil(np.array(load_Core.state_win_lengths)/win_len_sec).astype(np.int))
        end_ind = win_ind
        for le in num_winds:
            start_ind = np.max((0,end_ind-le))
            y[start_ind:end_ind] = state_counter
            state_counter +=1
            end_ind = start_ind
            if(end_ind<=0):
                break
        return y
        
    if(seizure_start_time_offsets>=0 and seizure_lengths>=0):
        for win_ind in range(num_windows):
            w = intervals_with_stride[win_ind]
            if((seizure_start_time_offsets<w + win_len) and (seizure_start_time_offsets + seizure_lengths >w)):
                y[win_ind] = y_detection
                if(not flag_ictal):
                    flag_ictal = True
                    y = state_gen(y, win_ind)
                        
    dim = X.shape[2]
    return X, y, S, dim, conv_sizes

    
