# In the name of God
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from GraphL_Utils import CoarseningParams, fine_mat_gen, MapFineToCoarse, plot_graph_density, SparsityCore, PlotCore
from GraphL_Utils import Fine2CoarseCore, FineMatCore, CoarseMatCore, sparsity_eval, Gsparsity_function
from GraphL_Utils import plot_array, FineCoarseMode, NormalDistProp, DeltaDistProp, SBMDistProp, UniformDistProp
from GraphL_Utils import MixtureDistProp, calc_all_sparsities, plot_line, plot_scatter, plot_regression
from GraphL_Utils import plot_graphScaling, plot_hist_text, plot_heat_text, graphPruning, plot_single_regression
from GraphL_Utils import arrayScaling, arrayShifting, arrayMaxForcing, arrayMinForcingScaling, select_fine_distribution
from GraphL_Utils import plot_heat, GenerateSBMLinCoarsMat, evalSparsityPreserving, SBMCommunityProbGenerator
from GraphL_Utils import sensingLayout, coverageRatio, SBMCore, geographical2DMesh, reorderMat, aggregateList
from GraphL_Utils import applyFuncPandas, SynchronizationCore
import pandas as pd
import networkx as nx
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)




V = 600 # [20, 50, 100, 200, 500, 1000] # 
N_array = [20] # list(np.arange(5, 200, 10)) # [200, 75, 25, 15] # [40] #  
coverage_array = [9]
dist_case_array = ['SBM']
fine_sample_size = 1
SBMLinCoars_sample_size = 1
SBM_Q_sample_size = 3

SchCompFlag = False
linearFlag = not SchCompFlag
lin_regularity_array = [1] # list(np.arange(0, 1, 0.1)) # [0.2] #   
overlap_array = [0] # list(np.arange(0, 0.9, 0.1)) #       
scaling = 1 # 'row_normalize'
graphCoarseningStr = 'Schur-Complement' if SchCompFlag else 'Linear'




df = pd.DataFrame(columns=['fine-size','fine-W', 'coarse-size','coarse-W'])
df[['fine-size', 'coarse-size']] = df[['fine-size', 'coarse-size']].astype('int') 

mesh = geographical2DMesh(V)
                    
for dist_case in dist_case_array:
    for N in N_array:
        for  coverage in coverage_array:
            for linCoasrseSampleCounter in np.arange(SBMLinCoars_sample_size):
                C = N                                
                print('------linear coarsening sample {}'.format(linCoasrseSampleCounter))
#                     sensing setup 
                LinCoarsMat_core = GenerateSBMLinCoarsMat(mesh, V, N, coverage, scaling)
                for SBMQ_SampleCounter in np.arange(SBM_Q_sample_size):
                    print('***linear coarsening sample {}'.format(SBMQ_SampleCounter))
#                   SBM setup
                    SBM_core = SBMCore(V, C, mesh, LinCoarsMat_core=LinCoarsMat_core, fixSyncRatio=True)
                    Sync_core = SynchronizationCore(LinCoarsMat_core, SBM_core)
                    
#                     coarsening setup
                    coarse_core = CoarseMatCore(size=N)
                    fine_coarse_mode = FineCoarseMode(Schur_comp=SchCompFlag, linear=linearFlag, scaling=scaling) 
                    fine2Coarse_core = Fine2CoarseCore(N, V, mode=fine_coarse_mode, linear_Coarsening_mat=LinCoarsMat_core.B)
                    fine_core = FineMatCore(size=V, distribution=SBMDistProp(SBM_core)) # select_fine_distribution(size=V , case=dist_case)
                    param = CoarseningParams(fine_core=fine_core, coarse_core=coarse_core, \
                                             fine2Coarse_core=fine2Coarse_core, sparsity_core=None)
                    
                    for fine_sample_counter in np.arange(fine_sample_size):
                        W = fine_mat_gen(fine_core)
                        W_tilde = MapFineToCoarse(W, param)
                        df_in = pd.DataFrame({'fine-size':[V], 'fine-W':[W],'coarse-size':[N],'coarse-W':[W_tilde], \
                                                    'SBM GraphSyncRatio': [Sync_core.syncRatio], 'SBM Qprobs': [SBM_core.Qprobs],\
                                                            'B': [LinCoarsMat_core.B], 'SBM comIndices': [aggregateList(SBM_core.comIndices)], \
                                                                'B repCommunities': [Sync_core.repCommunity]}) 
                        #TODO make if more efficient, not saving same W over and over
                        df = df.append(df_in) 
                
df['sensing layout'] = df['B'].apply(sensingLayout)
df['coverage ratio'] = df['B'].apply(coverageRatio)

if(True):
    df['reordered fine W'] = applyFuncPandas(df, reorderMat, sourceCol='fine-W', paramCol='SBM comIndices')
    df['reordered sensing layout'] = applyFuncPandas(df, reorderMat, sourceCol='sensing layout', paramCol='SBM comIndices')
#     df['reordered coarse W'], df['adjusted groundTruth Q SBM'] = applyFuncPandas(df, reorderAdjustCoarseMat, sourceCol='coarse-W', paramCol='SBM comIndices')
    
# df['coarse-fine-sparsity-preserving'] = applyFuncPandas(df, evalSparsityPreserving, sourceCol='reordered coarse W', paramCol='adjusted groundTruth Q SBM')

if(True):
    figName = 'example_SBM_graphCoarsening'.format()
    target_cols = ['SBM Qprobs', 'reordered fine W', 'reordered sensing layout', 'coarse-W']
    plot_heat(df, target_cols=target_cols, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                    saveFlag=True, showFlag=True, figsize = (3*len(target_cols), 2.4*df.shape[0]))) # 'Blues'

if(False):
    if(len(N_array)>1):
        x_col = 'coarse-size' 
    else:
        x_col = 'SBM GraphSyncRatio' #  ($\\pho$)
#     if(True):
#         groupby_col = 'full-fine-distribution'
#         df = df.melt(id_vars=[x_col,groupby_col], value_vars= ['mean','std'], value_name=groupby_col, var_name='whatever')
    target_cols = ['coarse-fine-sparsity-preserving']
    if(SchCompFlag):
        figName = 'sparsity_preserving_wrt_{}_{}_N={}_V={}'\
            .format(x_col, graphCoarseningStr, N_array[0], V)
    else:
        figName = 'sparsity_preserving_wrt_{}_{}_N={}_V={}_scaling={}'\
            .format(x_col, graphCoarseningStr, N_array[0], V, scaling)
    plot_line(df, x_col=x_col, target_cols=target_cols, groupby_col=None, plot_core=PlotCore(title='', figName=figName,\
                                                                           saveFlag=False, showFlag=True, log_scale=False))


