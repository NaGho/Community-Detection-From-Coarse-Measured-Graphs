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
from GraphL_Utils import applyFuncPandas, SynchronizationCore, communityRecovery, SyncCore, ElemkMeansRecovery
from GraphL_Utils import Row_Col_kMeansRecovery, PhiPermutationMap, SNAPnetworkGen, df_empty
from Community_utils import SSBMCore, measuringCore, CoarseningCommunityParams, evalCommunityRecovery
from Community_utils import recoverCommunities
import pandas as pd
import networkx as nx
from cdlib import algorithms, ensemble, evaluation, classes
from cdlib.classes.node_clustering import NodeClustering
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)

# B = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])
# B = B/np.sum(B, axis=1)[:,np.newaxis]
# P = np.array([[1,1,0,0,0,1],[0,0,1,1,1,0]])
# Wtilde = np.random.uniform(size=(3,3))
# comRec_core = communityRecovery(Wtilde, B, P) 
# comRec_core.relaxedOptimize(5) # comRec_core.fullOptimize()
# print('*$#@*',comRec_core.getSolution())


n = 10000 # [20, 50, 100, 200, 500, 1000] # 
L_array = [100] # list(np.arange(5, 200, 10)) # [200, 75, 25, 15] # [40] #  
K_array = [5]

alpha_array = [1.01, 1.5, 10] # np.arange(1, 52, 8) # 
beta = 1

nu = 2
coverage_array = [10]

dist_case = 'SBM'
fine_sample_size = 1
SBMLinCoars_sample_size = 1
networkGenMode = 'synthetic' # 'email-Eu-core' # 'SNAP-email-Eu-core'
scaling = 1 # 'row_normalize'

mesh = None # geographical2DMesh(n)
df = df_empty(['n','L', 'K', 'C'], ['int', 'int', 'int', 'int'])

for alpha in alpha_array:
    for L in L_array:
        for  C in coverage_array:
            for linCoasrseSampleCounter in np.arange(SBMLinCoars_sample_size):
                for K in K_array:                               
                    df_in = {'n':n, 'L':L, 'K':K, 'C':C, 'alpha':alpha, 'beta':beta, 'nu':nu, \
                             'tauTilde': 1e-4, 'p':alpha*np.log(n)/n, 'q':beta*np.log(n)/n}
                    measuring_core = measuringCore(df_in, GenerateSBMLinCoarsMat(mesh, n, L, C, scaling))                  
                    df_in['B'] = B = measuring_core.linCoarseningMat 
                    SSBM_core = SSBMCore(df_in, measuring_core, mesh)
                    param = CoarseningCommunityParams(df_in, graphGenMode=SBMDistProp(SSBM_core) if 'synthetic' in networkGenMode \
                                                                                else SNAPnetworkGen(networkGenMode), \
                                                                                measuring_core=measuring_core)
                    Phi = np.matmul(B, SSBM_core.P.T)
                    Phi_permutationMap = PhiPermutationMap(Phi)
                    df_in['true comIndices'] = [np.argwhere(Phi[:,k]>0).T[0].tolist() for k in np.arange(K)] # (Phi>0).astype(int) # [aggregateList(SSBM_core.comIndices)]
                    for fine_sample_counter in np.arange(fine_sample_size):
                        W = param.graphGenMode.sample(n)
                        W_tilde = np.matmul(np.matmul(B,W), B.T) # MapFineToCoarse(W, param)
                        df_in['recovered comIndices'] = recoverCommunities(df_in, W_tilde)
                        
                        #TODO make if more efficient, not saving same W over and over
                        df_in['fine-W'] = W
                        df_in['coarse-W'] = W_tilde
                        df_in['Community Recovery Error'] = evalCommunityRecovery(df_in, df_in['recovered comIndices'], df_in['true comIndices'])
                        df = df.append(df_in, ignore_index=True) 
                    
df['sensing layout'] = df['B'].apply(sensingLayout)
df['coverage ratio'] = df['B'].apply(coverageRatio)

if(True):
    df['reordered fine W'] = applyFuncPandas(df, reorderMat, sourceCol='fine-W', paramCol='true comIndices')
    df['reordered sensing layout'] = applyFuncPandas(df, reorderMat, sourceCol='sensing layout', paramCol='true comIndices')
   
    
# df['Community Recovery Error'] = applyFuncPandas(df, evalCommunityRecovery, sourceCol='recovered comIndices', paramCol='true comIndices')

if(False):
    df_new = pd.DataFrame()
    for i in np.arange(int(df.shape[0]/2)):
        df_in = df.iloc[i*2,:]
        df_in['Community Recovery Error (known)'] = df_in['Community Recovery Error'] 
        df_in['Recovered Q (known)'] = df_in['Recovered Q']
        df_in['Community Recovery Error (unknown)'] = df['Community Recovery Error'].iloc[i+1]
        df_in['Recovered Q (unknown)'] = df['Recovered Q'].iloc[i*2+1]
        df_new = df_new.append(df_in)
    figName = 'example_SBM_graphCoarsening_SBMdensity={}_C={}_N={}_V={}_synchRatio={}'.format(SBM_Qdensity, C_array[0], L_array[0], n, proto_core.syncRatio)
    target_cols = ['True Q', 'fine-W', 'reordered fine W', 'reordered sensing layout', 'coarse-W', 'Recovered Q (known)', 'Recovered Q (unknown)']
    plot_heat(df_new, target_cols=target_cols, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                    saveFlag=True, showFlag=True, figsize = (3.5*len(target_cols), 2.8*df_new.shape[0]))) # 'Blues'

if(False):
#     x_col = 'coarse-size' 
    x_col = 'Synchronization Ratio' #  ($\\pho$)
#     if(True):
#         groupby_col = 'full-fine-distribution'
#         df = df.melt(id_vars=[x_col,groupby_col], value_vars= ['mean','std'], value_name=groupby_col, var_name='whatever')
    target_cols = ['Community Recovery Error']
    figName = 'Community_Recovery_Error_wrt_{}_{}_N={}_V={}'.format(x_col, graphCoarseningStr, L_array[0], n)
    plot_line(df, x_col=x_col, target_cols=target_cols, groupby_col=None, plot_core=PlotCore(title='', figName=figName,\
                                                                           saveFlag=False, showFlag=True, log_scale=False))

if(False):
    x_col = 'Synchronization Ratio'
    y_col = 'Community Recovery Error'
    figName = 'Community_Recovery_Error_wrt_{}_{}_N={}_V={}_sync_calc_mode={}'.format(x_col, graphCoarseningStr, L_array[0], n, sync_calc_mode)
    plot_single_regression(df, x_col=x_col, y_col=y_col, hue_col='Recovery Regime',\
                                plot_core=PlotCore(title='', figName=figName,\
                                   saveFlag=True, showFlag=True, log_scale=False))

