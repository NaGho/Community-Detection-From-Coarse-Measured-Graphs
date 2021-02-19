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
from GraphL_Utils import Row_Col_kMeansRecovery, PhiPermutationMap, SNAPnetworkGen
import pandas as pd
import networkx as nx
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


V = 900 # [20, 50, 100, 200, 500, 1000] # 
N_array = [30] # list(np.arange(5, 200, 10)) # [200, 75, 25, 15] # [40] #  
C_array = [20]
SBM_Qdensity = 0.14
coverage_array = [5]
dist_case = 'SBM'
fine_sample_size = 1
SBMLinCoars_sample_size = 1
SBM_Q_sample_size = 3
sync_calc_mode = 'max' # 'max',  'median'
networkGenMode = 'email-Eu-core' # 'synthetic', 'SNAP-email-Eu-core'

SchCompFlag = False
linearFlag = not SchCompFlag
lin_regularity_array = [1] # list(np.arange(0, 1, 0.1)) # [0.2] #   
overlap_array = [0] # list(np.arange(0, 0.9, 0.1)) #       
scaling = 'row_normalize'
graphCoarseningStr = 'Schur-Complement' if SchCompFlag else 'Linear'




df = pd.DataFrame(columns=['fine-size','fine-W', 'coarse-size','coarse-W'])
df[['fine-size', 'coarse-size']] = df[['fine-size', 'coarse-size']].astype('int') 

syncVals = [1/4, 1/2, 3/4, 1]
maxCommunityCoverage = 3
if(True):
    syncProb_array =  [np.array([0, 0.2, 0.2, 0.6])] # [0, 0, 0, 1] #  [0.1, 0.3, 0.4, 0.2] # 
else:
    syncProb_array = [np.array([0, 0, 0, 1])]
    syncProb_array.extend([np.array([0, 0, 0.1, 0.9]), np.array([0, 0, 0.2, 0.8])])
    syncProb_array.extend([np.sort(np.random.uniform(low=0, high=100, size=4)) for _ in np.arange(10)]) 
    syncProb_array.extend([np.random.uniform(low=0, high=100, size=4) for _ in np.arange(5)])
    syncProb_array.extend([-np.sort(-np.random.uniform(low=0, high=100, size=4)) for _ in np.arange(5)])
    syncProb_array.extend([np.concatenate(([1], np.random.uniform(low=0, high=0.2, size=2), [0])) for _ in np.arange(10)])

syncProb_array = [x/x.sum() for x in syncProb_array]

mesh = geographical2DMesh(V)

for N in N_array:
    for  coverage in coverage_array:
        for linCoasrseSampleCounter in np.arange(SBMLinCoars_sample_size):
            for C in C_array:                               
#                     sensing setup 
                LinCoarsMat_core = GenerateSBMLinCoarsMat(mesh, V, N, coverage, scaling)
                for syncProb in syncProb_array:
                    for SBMQ_SampleCounter in np.arange(SBM_Q_sample_size):
                    
    #                   SBM setup
                        Sync_core = SyncCore(N, C, syncVals=syncVals, syncProb=syncProb, maxCommunityCoverage=maxCommunityCoverage)
                        try:
                            SBM_core = SBMCore(V, C, mesh, density=SBM_Qdensity, LinCoarsMat_core=LinCoarsMat_core, setupSync_core=Sync_core)
                        except:
                            continue
                        Sync_core.update(LinCoarsMat_core, SBM_core, sync_calc_mode=sync_calc_mode)
    #                         Sync_core = SynchronizationCore(LinCoarsMat_core, SBM_core)
                        
    #                     coarsening setup
                        coarse_core = CoarseMatCore(size=N)
                        fine_coarse_mode = FineCoarseMode(Schur_comp=SchCompFlag, linear=linearFlag, scaling=scaling) 
                        fine2Coarse_core = Fine2CoarseCore(N, V, mode=fine_coarse_mode, linear_Coarsening_mat=LinCoarsMat_core.B)
                        fine_core = FineMatCore(size=V, graphGenMode=SBMDistProp(SBM_core) if 'synthetic' in networkGenMode \
                                                                                    else SNAPnetworkGen(networkGenMode)) 
                        # select_fine_distribution(size=V , case=dist_case)
                        param = CoarseningParams(fine_core=fine_core, coarse_core=coarse_core, \
                                                 fine2Coarse_core=fine2Coarse_core, sparsity_core=None)
                        
                        comRec_core = communityRecovery(LinCoarsMat_core.B, SBM_core.P) 
                        Phi = np.matmul(LinCoarsMat_core.B, SBM_core.P.T)
                        Phi_permutationMap = PhiPermutationMap(Phi)
                        
    #                         np.matmul(LinCoarsMat_core.B, SBM_core.P.T)
                        # np.any(LinCoarsMat_core.B!=SBM_core.P/np.sum(SBM_core.P,axis=1)[:,np.newaxis])
                        for fine_sample_counter in np.arange(fine_sample_size):
                            W = fine_mat_gen(fine_core)
                            W_tilde = MapFineToCoarse(W, param)
                            if(True):
                                Q_hat_unknown = Row_Col_kMeansRecovery(W_tilde, SBM_core.C)
                                comRec_core.relaxedOptimize(W_tilde, 0.01)
                                Q_hat_known = comRec_core.getSolution()
                            else:
                                Q_hat = ElemkMeansRecovery(W_tilde, SBM_core.C)
                                
                            
                            df_in1 = pd.DataFrame({'fine-size':[V], 'fine-W':[W],'coarse-size':[N],'coarse-W':[W_tilde], \
                                        'Synchronization Ratio': [Sync_core.syncRatio], 'True Q': [SBM_core.Qprobs],\
                                            'B': [LinCoarsMat_core.B], 
                                                'Recovery Regime':['known'],\
                                                    'Recovered Q': [Q_hat_known],\
                                                        'SBM comIndices': [aggregateList(SBM_core.comIndices)]}) 
                            df_in2 = df_in1.copy()
                            df_in2['Recovery Regime'] = ['unknown']
                            df_in2['Recovered Q'] = [reorderMat(Q_hat_unknown, fromPermutation=Phi_permutationMap)]
                            #TODO make if more efficient, not saving same W over and over
                            df = df.append(df_in1) 
                            df = df.append(df_in2) 
                            
df['sensing layout'] = df['B'].apply(sensingLayout)
df['coverage ratio'] = df['B'].apply(coverageRatio)

if(True):
    df['reordered fine W'] = applyFuncPandas(df, reorderMat, sourceCol='fine-W', paramCol='SBM comIndices')
    df['reordered sensing layout'] = applyFuncPandas(df, reorderMat, sourceCol='sensing layout', paramCol='SBM comIndices')
#     df['reordered coarse W'], df['adjusted groundTruth Q SBM'] = applyFuncPandas(df, reorderAdjustCoarseMat, sourceCol='coarse-W', paramCol='SBM comIndices')
    
    
    
df['Community Recovery Error'] = applyFuncPandas(df, evalSparsityPreserving, sourceCol='Recovered Q', paramCol='True Q')

if(False):
    df_new = pd.DataFrame()
    for i in np.arange(int(df.shape[0]/2)):
        df_in = df.iloc[i*2,:]
        df_in['Community Recovery Error (known)'] = df_in['Community Recovery Error'] 
        df_in['Recovered Q (known)'] = df_in['Recovered Q']
        df_in['Community Recovery Error (unknown)'] = df['Community Recovery Error'].iloc[i+1]
        df_in['Recovered Q (unknown)'] = df['Recovered Q'].iloc[i*2+1]
        df_new = df_new.append(df_in)
    figName = 'example_SBM_graphCoarsening_SBMdensity={}_C={}_N={}_V={}_synchRatio={}'.format(SBM_Qdensity, C_array[0], N_array[0], V, Sync_core.syncRatio)
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
    figName = 'Community_Recovery_Error_wrt_{}_{}_N={}_V={}'.format(x_col, graphCoarseningStr, N_array[0], V)
    plot_line(df, x_col=x_col, target_cols=target_cols, groupby_col=None, plot_core=PlotCore(title='', figName=figName,\
                                                                           saveFlag=False, showFlag=True, log_scale=False))

if(True):
    x_col = 'Synchronization Ratio'
    y_col = 'Community Recovery Error'
    figName = 'Community_Recovery_Error_wrt_{}_{}_N={}_V={}_sync_calc_mode={}'.format(x_col, graphCoarseningStr, N_array[0], V, sync_calc_mode)
    plot_single_regression(df, x_col=x_col, y_col=y_col, hue_col='Recovery Regime',\
                                plot_core=PlotCore(title='', figName=figName,\
                                   saveFlag=True, showFlag=True, log_scale=False))

