# In the name of God
import numpy as np
import json
import os,sys,inspect
from GraphL_Utils import SignalGen, signal2Graph, graphGen_PostProcessing, plot_graph_density
from GraphL_Utils import MapFineToCoarse, CoarseningParams, SparsityCore, plot_heat_hist_text
from GraphL_Utils import plot_array, PlotCore, LpSparsity, GiniIndex, HoyerSparsity
from GraphL_Utils import minusLogGdisparity, k4Sparsity, l2l1Sparsity, plot_boxplot, LeSparsity
from GraphL_Utils import plot_regression, calc_all_sparsities, minMaxNormalize, graphPruning
import random
from scipy import signal
from supervised_tasks import LoadCore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(precision=2)


# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 
# from final_graphL_feat_class.supervised_tasks import LoadCore, data_load, PreProcessing
# sys.path.insert(0,currentdir) 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""




class TaskCore(object):
    def __init__(self, data_dir, settings_dir, target, signal_mode, graphL_model=None, sidinfo_dir=None, \
                 cv_ratio=None, adj_calc_mode=None, TrainTest_mode=None, supervised=None,\
                 filters=None, num_szr_samp=None, num_preszr_samp=None, num_nonszr_samp=None):
        self.data_dir = data_dir
        self.settings_dir = settings_dir
        self.target = target
        self.graphL_model = graphL_model
        self.signal_mode = signal_mode
        self.sidinfo_dir = sidinfo_dir
        self.cv_ratio = cv_ratio
        self.adj_calc_mode = adj_calc_mode
        self.TrainTest_mode = TrainTest_mode
        self.supervised = supervised
        self.filters = filters
        self.num_szr_samp = num_szr_samp
        self.num_preszr_samp = num_preszr_samp
        self.num_nonszr_samp = num_nonszr_samp
        
        
class Filter(object):    
    def __init__(self, fs, lowcut, highcut): # all in Hz    
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        
# Initializations
KernelBW = 0.01
plotting_flag = False
with open('SETTINGS.json') as f:
    settings = json.load(f)
targets = [253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970, \
                        1084, 1096, 1146, 115, 139, 442, 635, 818, 1073, 1077, 1125, 1150, 732, 13089, 13245]
# targets = [620, 1096]
targets = [620]
graphL_models = ['corr'] #  ----- 'corr', 'cov', 'coherence', 'invCov', 'cross-spectrum'
signal_mode = 'ECoG' #  'ECoG', 'toy' 
# [1, 4] Hz, [5, 8] Hz (theta), [9, 13] Hz (alpha), [14, 25] Hz (beta), [25, 90] Hz (gamma), and [100, 200] Hz (Table 2)
lowCuts = [1, 5, 9, 14, 25, 100]
highCuts = [4, 8, 13, 25, 90, 200]
bandSelSrt = ['1-4', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma']
bandNumSelect = 3
load_core = LoadCore(preszr_sec=10, postszr_sec=10000, band_nums=None) #, idx_szr=[2604,1272], idx_nonszr=[459,4329])
task_core = TaskCore(data_dir=str(settings['data-dir-Raw']), \
                     settings_dir=str(settings['settings-dir-Raw']),\
                     target=targets[0], signal_mode=signal_mode, \
                     filters=[Filter(256, lowCuts[bandNumSelect], highCuts[bandNumSelect])],\
                     num_szr_samp=1, num_preszr_samp=0, num_nonszr_samp=0)
#                      num_szr_samp=None, num_preszr_samp=None, num_nonszr_samp=None) 
cloningInvFlag = True

# str(settings['data-dir-FFT']) if 'coherence' in graphL_model else 
# str(settings['settings-dir-FFT']) if 'coherence' in graphL_model else 

# Sample rate and desired cutoff frequencies (in Hz). fs = 5000.0  --- lowcut = 500.0  ---   highcut = 1250.0

df = pd.DataFrame(columns=['Patient','# Nodes','label','W'])
for target in targets:
    # Data Loading
    task_core.target = target
    try:
        X, y, sig_idx = SignalGen(task_core).run(load_core)
    except:
        print('Error while Loading Patient {}'.format(target))
        continue
    print('X shape: ', X.shape)
    for graphL_model in graphL_models:
        task_core.graphL_model = graphL_model         
        # Graph Learning
        W = signal2Graph(X, task_core)
        print('W shape: ', W.shape)
        if(False):
            W = graphGen_PostProcessing(W)
            
        df_in = pd.DataFrame({'Patient': target, 'N': W.shape[-1]*np.ones_like(y), 'label': y, \
                                'W': [np.squeeze(W[i,:,:]) for i in np.arange(W.shape[0])],\
                                'graphL_model': graphL_model})
        sparsity_core, df_in = calc_all_sparsities(df_in, KernelBW, columnToApply='W', cloningInvFlag = cloningInvFlag)
        if(True):
            groupby_col = 'label'
            target_cols = ['Gsparsity', 'SparsityL0', 'SparsityL1', 'SparsityLe', 'GiniIndex', 'Hoyer', \
                           'minusLogGdisparity', 'k4Sparsity', 'l2l1Sparsity']
            plot_boxplot(df_in, groupby_col=groupby_col, target_cols=target_cols, \
                                        plot_core= PlotCore(title='', figName='EcoG_{}{}_PT{}_Sparsities_comp_Bar'.\
                                            format(graphL_model, bandSelSrt[bandNumSelect] if 'coherence' in graphL_model else '',target),\
                                            saveFlag=False, showFlag=True, log_scale=False))
        df = df.append(df_in)
    
# ------------------------------------------------------------------------------------------------------------    
# sorting
# W_arr = [x for _,x in sorted(zip(N_array, W_arr))]
# y_array = np.array([x for _,x in sorted(zip(N_array, y_array))])
# idx_szr = np.argwhere(y_array>0)
# ------------------------------------------------------------------------------------------------------------

df['normalized-W'] = df['W'].apply(minMaxNormalize)
sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='W', cloningInvFlag = cloningInvFlag)
if(False):
    groupby_col = '# Nodes'
    target_cols = ['Gsparsity', 'L0Norm', 'L1Norm', 'LeNorm', 'GiniIndex', 'HoyerSparsity', 'minusLoGsparsity', 'k4Sparsity', 'l2l1Sparsity']
    plot_regression(df, groupby_col=groupby_col, target_cols=target_cols, \
                                plot_core=PlotCore(title='', figName='EcoG_{}{}_Gsparsity_N_regression'.\
                                                   format(graphL_model, bandSelSrt[bandNumSelect] if 'coherence' in graphL_model else ''),\
                                   saveFlag=False, showFlag=True, log_scale=False))
elif(False):
    plot_graph_density(df['W'].values, extra_heatMat_arr=None, \
                                plot_core=PlotCore(sparsity_core=sparsity_core, \
                                    log_scale=False, match_ranges=False, figsize=(10,20),\
                                        num_bins=None, showFlag=True, saveFlag=False, \
                                            figName='EcoG_sample_{}_HeatMap_Histogram_{}_szr-idx={}_nonszr-idx={}'.\
                                                format(graphL_model, bandSelSrt[bandNumSelect] if 'coherence' in graphL_model else '',\
                                                        '-'.join(str(x) for x in sig_idx['idx_szr']),\
                                                        '-'.join(str(x) for x in sig_idx['idx_nonszr']))))
elif(False):
    plot_graph_density(df['normalized-W'].values, extra_heatMat_arr=None, \
                                plot_core=PlotCore(sparsity_core=sparsity_core, \
                                    log_scale=False, match_ranges=False, figsize=(10,15),\
                                        num_bins=None, showFlag=True, saveFlag=False, \
                                            figName='EcoG_sample_{}_HeatMap_Histogram_{}_szr-idx={}_nonszr-idx={}'.\
                                                format(graphL_model, bandSelSrt[bandNumSelect] if 'coherence' in graphL_model else '',\
                                                        '-'.join(str(x) for x in sig_idx['idx_szr']),\
                                                        '-'.join(str(x) for x in sig_idx['idx_nonszr']))))
    
## Intoduction Pruning Comparison
elif(False):
    figName = 'intro_pruning_notNormalized'.format()
    cols_str = ['W'] 
    col_counter = 0
    
    colStr = 'fixed edgeDensity=0.2'
    df[colStr] = df['normalized-W'].apply(graphPruning, args=[None, 0.2])
    cols_str.append(colStr)
    col_counter += 1
    
    threshold = 0.3 # np.mean(np.abs(df['W'].iloc[0]))
    colStr = 'fixed threshold={0:.1f}'.format(threshold)
    df[colStr] = df['W'].apply(graphPruning, args=[threshold, None]) # df[target_col].iloc[0].max()
    cols_str.append(colStr)
    col_counter += 1
    
#     threshold = int(np.floor(np.abs(np.mean(df['W'].iloc[1]))))
#     colStr = 'fixed threshold={}'.format(threshold)
#     df[colStr] = df['W'].apply(graphPruning, args=[threshold, None]) # df[target_col].iloc[0].max()
#     cols_str.append(colStr)
#     col_counter += 1
    
    for col_str in cols_str:
        _, df = calc_all_sparsities(df, KernelBW, columnToApply=col_str, ColExtStr=col_str+'-', sparsity_core=sparsity_core)
    
    
    plot_heat_hist_text(df, target_cols=cols_str,\
                        text_cols=[[],\
                                   ['SparsityL0','SparsityL1','Hoyer','GiniIndex'],\
                                   []],\
                            plot_core=PlotCore(title='', figName=figName, match_ranges=False, saveFlag=False, showFlag=True, figsize = (16, 5), palette_cmap=None),\
                                heat_flags=[True, True, True], hist_flags=[True, False, False],\
                                    text_flags=[False, True, False],\
                                        add_remove_text_cols=cols_str)
#     -----------------------------------------------------------------------------------------------------
    figName = 'pruning_Gsparsity'.format()
    cols_str = ['normalized-W'] 
    col_counter = 0
    
    colStr = 'fixed edgeDensity=0.2'
    df[colStr] = df['normalized-W'].apply(graphPruning, args=[None, 0.2])
    cols_str.append(colStr)
    col_counter += 1
    
#     colStr = 'fixed threshold=0.1'
#     df[colStr] = df['normalized-W'].apply(graphPruning, args=[0.1, None])
#     cols_str.append(colStr)
#     col_counter += 1
    colStr = 'fixed threshold=0.3'
    df[colStr] = df['normalized-W'].apply(graphPruning, args=[0.2, None])
    cols_str.append(colStr)
    col_counter += 1
    
#     colStr = 'edgeDensity=1-Gsparsity'
#     df[colStr] = [graphPruning(df['normalized-W'].iloc[i], None, 1-df['Gsparsity'].iloc[i])  for i in np.arange(df['normalized-W'].size)]
#     cols_str.append(colStr)
#     col_counter += 1
    
    for col_str in cols_str:
        _, df = calc_all_sparsities(df, KernelBW, columnToApply=col_str, ColExtStr=col_str+'-', sparsity_core=sparsity_core)
    
    
    plot_heat_hist_text(df, target_cols=cols_str,\
                        text_cols=[[],\
                                   ['SparsityL0','SparsityL1','Hoyer','GiniIndex'],\
                                   []],\
                            plot_core=PlotCore(title='', figName=figName, match_ranges=False, saveFlag=False, showFlag=True, figsize = (16, 5), palette_cmap=None),\
                                heat_flags=[True, True, True], hist_flags=[True, False, False],\
                                    text_flags=[False, True, False],\
                                        add_remove_text_cols=cols_str) # 'Blues'

# ------------------------------------------------------------------------------------------------------------
