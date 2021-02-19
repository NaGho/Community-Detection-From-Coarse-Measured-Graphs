import numpy as np
import matplotlib
# matplotlib.use('GTK')  # Or any other X11 back-end
import matplotlib.pyplot as plt
import seaborn as sns
from GraphL_Utils import CoarseningParams, fine_mat_gen, MapFineToCoarse, plot_graph_density, SparsityCore, PlotCore
from GraphL_Utils import Fine2CoarseCore, FineMatCore, CoarseMatCore, sparsity_eval, Gsparsity_function
from GraphL_Utils import plot_array, FineCoarseMode, NormalDistProp, DeltaDistProp, SBMDistProp, UniformDistProp
from GraphL_Utils import MixtureDistProp, calc_all_sparsities, plot_line, plot_scatter, plot_regression
from GraphL_Utils import plot_graphScaling, plot_hist_text, plot_heat_text, graphPruning, plot_single_regression
from GraphL_Utils import arrayScaling, arrayShifting, arrayMaxForcing, arrayMinForcingScaling, select_fine_distribution
from GraphL_Utils import plot_heat, evalSparsityPreserving, SBMCommunityProbGenerator, relPlot
from GraphL_Utils import sensingLayout, coverageRatio, SBMCore, geographical2DMesh, reorderMat, aggregateList
from GraphL_Utils import applyFuncPandas, SynchronizationCore, communityRecovery, SyncCore, ElemkMeansRecovery
from GraphL_Utils import Row_Col_kMeansRecovery, PhiPermutationMap, SNAPnetworkGen, df_empty, findLargerDivisible
# import GraphL_Utils as GraphL_Utils
from Community_utils import SSBMCore, measuringCore, CoarseningCommunityParams, evalCommunityRecovery
from Community_utils import recoverCommunities, GenerateSBMLinCoarsMat, mat2List, min_coverageSBM
from Community_utils import UB_Failure_Error_Function
import pandas as pd
import networkx as nx
from cdlib import algorithms, ensemble, evaluation, classes
from cdlib.classes.node_clustering import NodeClustering
# from numba import jit, cuda
# from timeit import default_timer as timer
from datetime import datetime
# import graph_tool.all as gt
# from graph_tool import spectral
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)

'''
 ErrorBoundFig 
 ---------------------
 wrt L for gap:
 n_array = [30000], K_array = [5], nu_array = [2], fine_sample_size = 2, m_array = [10, 20, 50, 100, 150, 200, 300, 400, 500], 
 alpha_array = [500, 2000], beta = 50, coverage_array = [50]
---------------------
wrt C for nu
 n_array = [30000], K_array = [5], nu_array = [2, 3], fine_sample_size = 2, m_array = [300], 
 alpha_array = [500], beta = 50, coverage_array = [80, 70, 60, 50, 40, 30, 20, 10, 5, 2]
---------------------
wrt n for K:
n_array = np.multiply([5, 6, 7, 8, 9, 10, 11, 12], 1e6), K_array = [4], nu_array = [2, 3], m_array = [1000], coverage_array = [2000]
alpha_array = [10000], beta = 1000

---------------------
Synthetic Graph Community Recovery:


'''
# Test graph-tool module for SBM 
if(False):
    g = gt.collection.data["polblogs"]
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    g = gt.Graph(g, prune=True)
    state = gt.minimize_blockmodel_dl(g)
    # print('state.b.a = ', state.b.a)
    # print('state.get_bg() = ', state.get_bg())
    # print('state.get_ers() = ', state.get_ers()) # two-dim numpy array showing the average weight between pairs
    # print('get.adjacency = {}  \n ******'.format(gt.adjacency(state.get_bg(), state.get_ers()).T))
    u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(), state.get_ers()).T)
    
windows = True
Community_Recover = True
ErrorBound_Calc = True
n_array = [3000] # np.multiply([5, 6, 7, 8, 9, 10, 11, 12], 1e6) # [20, 50, 100, 200, 500, 1000] # 
K_array = [5] # [4] # n should be divisible by K
nu_array = [2] # C should be divisible by nu
m_array = [30] # [10, 20, 50, 100, 150, 200, 300, 400, 500] # [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800] # [40, 50] # list(np.arange(5, 200, 10)) # [200, 75, 25, 15] # [40] #  coverage_array = [1000]


dist_case = 'SBM'
fine_sample_size = 2
# SBMLinCoars_sample_size = 1
networkGenMode = 'synthetic' # 'com-youtube.ungraph' # 'email-Eu-core' # 
genModule = 'networkX' # 'networkX' # 'graph-tool' # 
scaling = 1 # 'row_normalize'
 
mesh = None # geographical2DMesh(n)


def main():
    df = df_empty(['n','m', 'K', 'r', 'Failure UB Error'], ['int', 'int', 'int', 'int', 'float'])  
    for n in n_array:
        alpha_array = [50] # np.multiply([0.3], n/np.log(n)) # np.arange(1, 52, 8) # 
        beta = 5 # 0.05 * n/np.log(n)
        for alpha in alpha_array:
            for K in K_array:        
              for nu in nu_array:  
                  tauTilde_array = [1/(nu+1)] # should be strictly greater than 0 and strictly less than 1/nu
                  for tauTilde in tauTilde_array:
                    fine_sample_counter = 0     
                    while(fine_sample_counter<fine_sample_size):
                        start_time = datetime.now()
                        if('synthetic' in networkGenMode):
                            p = alpha*np.log(n)/n
                            q = beta*np.log(n)/n
                            print('p={}, q={} '.format(p, q))
                            if(p>1 or q>1 or p<0 or q<0):
                                print('alpha-beta invalid!')
                                continue
                            df_in = {'n':n, 'K':K, 'alpha':alpha, 'beta':beta, 'nu':nu, 'tauTilde': tauTilde, 'scaling':scaling,\
                                     'alpha-beta Gap':(alpha+beta)/2-np.sqrt(alpha*beta), 'p':p, 'q':q}
                            minC = min_coverageSBM(df_in)
                            if(True):
                                print('Minimum Coverage size = ', minC)
                            if(Community_Recover):
                                SSBM_core = SSBMCore(df_in, measuring_core=None, mesh=None, genModule=genModule)
                                param = CoarseningCommunityParams(df=None, graphGenMode=SBMDistProp(SSBM_core), measuring_core=None)                        
                                fine_graph, _ = param.graphGenMode.sample(n, df_in)   
                                if(False):
                                    print('time elapsed generating: {} sec'.format((datetime.now()-start_time).seconds))
                                # r should be divisible by 2, ..., nu # m*r<n & n/Kr better be multiples of {1, 1/2, ..., 1/nu} for rm close to n
                                # r<< n/K 
                                coverage_array = [40] # [findLargerDivisible(minC, nu)] # [2, 4, 6, 8] 
                        else:
                            df_in = {'nu':nu, 'tauTilde': tauTilde, 'scaling':scaling}
                            if(Community_Recover):
                                param = CoarseningCommunityParams(df=None, graphGenMode=SNAPnetworkGen(networkGenMode), measuring_core=None)   
                                fine_graph, df_in = param.graphGenMode.sample(n, df_in)   
                                if(True):
                                    print('time elapsed loading: {} sec'.format((datetime.now()-start_time).seconds))
                                # r should be divisible by 2, ..., nu
                                coverage_array = [df_in['minCommunitySize']]
                                    
                        true_comIdx_fine = fine_graph.communityPartitions
                        lenComs = [len(idx) for idx in true_comIdx_fine]
                        print('Length of true communities = {}, sum to ={}, n={}'.format(lenComs, np.sum(lenComs), n))
                        if(np.sum(lenComs)!=n):
                            continue
                        for r in coverage_array:
                            r = findLargerDivisible(r, nu)
                            if(False and r<min_r):
                                print('r<minimum r')
                                continue
                            for m in m_array:
                                print('r={}'.format(r))
                                if(r*m>n):
                                    print('Invalid Values, r*m>n!')
                                    continue
                                df_in['r'] = r
                                df_in['m'] = m
                                if(ErrorBound_Calc):
                                    try:
                                        error_bound, df_in = UB_Failure_Error_Function(df_in)
                                        if(not Community_Recover):
                                            df = df.append(df_in, ignore_index=True)
                                    except:
                                        raise ValueError('Error while Computing theoretical upper bound!')
                                    if(True):
                                        print('error bound = ', error_bound)
                                if(Community_Recover):
                                    try:
                                        start_time = datetime.now()
                                        measuring_core = measuringCore(df_in, mesh=None, true_comIdx_fine=true_comIdx_fine)
                #                         B = measuring_core.linCoarseningMat 
                                        W_tilde = MapFineToCoarse(fine_graph, measuring_core=measuring_core) # np.matmul(np.matmul(B,W), B.T) # 
                                        print('time elapsed measurement: {} sec'.format((datetime.now()-start_time).seconds))
                                        true_comIdx_coarse = mat2List(measuring_core.normProfileMat)
    #                                     df['min Community Size'] = np.min([len(idx) for idx in true_comIdx_coarse])
                                        start_time = datetime.now()
                                        recovered_comIdx = recoverCommunities(df_in, W_tilde, true_comIdx_coarse=true_comIdx_coarse) # df_in['recovered comIndices'] = 
                                        if(False):
                                            print('partitioning result: ', recovered_comIdx)
                                        #TODO make if more efficient, not saving same W over and over
    #                                     df_in['fine-W'] = W
    #                                     df_in['coarse-W'] = W_tilde
                                        if(True):
                                            print('GroundTruth partitioning: ', true_comIdx_coarse)
                                            for method, idx in recovered_comIdx.items():
                                                print('Recovered partitioning for {} is: {}'.format(method, idx))
                                        df_in.update(evalCommunityRecovery(df_in, recovered_comIdx, true_comIdx_coarse, W_tilde))
    #                                     df_in['Community Recovery Error'] = 
                                        df = df.append(df_in, ignore_index=True) 
                                        print(df)
                                        if(False):
                                            df.to_csv('out.csv', index=False) 
                                        if(True):
                                            print('NF1 Recovery Error with M-NMF = ', df_in['NF1 Recovery Error with M-NMF'])
                                            print('time elapsed recovery: {} sec'.format((datetime.now()-start_time).seconds))
                                    except Exception as exc:
                                        print(exc)
                                        continue
                        fine_sample_counter += 1
    #         df['sensing layout'] = df['B'].apply(sensingLayout)
    #         df['coverage ratio'] = df['B'].apply(coverageRatio)

        
    if(False):
        df['reordered fine W'] = applyFuncPandas(df, reorderMat, sourceCol='fine-W', paramCol='true comIdx')
        df['reordered sensing layout'] = applyFuncPandas(df, reorderMat, sourceCol='sensing layout', paramCol='true comIdx')
   
    
    # df['Community Recovery Eval'] = applyFuncPandas(df, evalCommunityRecovery, sourceCol='recovered comIndices', paramCol='true comIndices')
    return df


if(True):
    df = main()
else:
    df = pd.read_csv('out_wrt_coverage_for_nu.csv') # 'out_wrt_measurementSize_for_p.csv' , 'out_wrt_coverage_for_nu.csv'
    
print(df[['n', 'm', 'r', 'K']].join(df.loc[:, df.columns.str.contains('Recovery Error')]))

value_cols = ['Failure UB Error', 'Failure UB Error (Lowest U)','Failure UB Error (Highest U)']
df = df.melt(id_vars=list(set(df.columns).difference(set(value_cols))), value_vars=value_cols, \
                                    value_name='Failure UB Error', var_name='Low-Mean-High UB Error')




     
if(Community_Recover):
    x_col = 'm' # 'n' # 'r' # 'm' # 'K' # 'nu' # 'tauTilde' # 'alpha'
    y_col = 'Community Recovery Error'
    hue_col = 'K'
    title = 'CommunityRecoveryError'
    figName = title + '_wrt_{}_for{}_n{}_m{}_K{}_nu{}'.format(x_col, hue_col, n_array[0], m_array[0], K_array[0], nu_array[0])
    if(not windows):
        df.to_csv(figName+'.csv', index=False) # df = pd.read_csv(figName+'.csv')
    plot_single_regression(df, x_col=x_col, y_col=y_col, hue_col=hue_col, plot_core=PlotCore(title='', figName=figName,\
                                   saveFlag=not windows, showFlag=windows, log_scale=True, minX_axis=0, maxX_axis=1))
    


elif(ErrorBound_Calc):
    x_col = 'r' # 'r' # 'n' # 'm' # 'K' # 'nu' # 'tauTilde' # 'ML Failure UB Error'
    hue_col = 'alpha-beta Gap' # 'nu'
    hue_col2 = None
#     if(True):
#         groupby_col = 'full-fine-distribution'
#         df = df.melt(id_vars=[x_col,groupby_col], value_vars= ['mean','std'], value_name=groupby_col, var_name='whatever')
    target_cols = ['Failure UB Error']
    title = 'UB_Failure_Error'
    figName = title + '_wrt_{}_n{}_m{}_K{}_nu{}'.format(x_col, n_array[0], m_array[0], K_array[0], nu_array[0])
    plot_line(df, x_col=x_col, target_cols=target_cols, hue_col=hue_col, hue_col2=hue_col2,\
                                                         plot_core=PlotCore(title='', figName=figName, saveFlag=not windows, \
                                                                    showFlag=windows, log_scale=True,  minX_axis=0, maxX_axis=1))


    
if(False):
    figName = 'example_SBM_graphCoarsening_SBMdensity={}_C={}_N={}_V={}_synchRatio={}'.format(SBM_Qdensity, r_array[0], m_array[0], n, proto_core.syncRatio)
    target_cols = ['True Q', 'fine-W', 'reordered fine W', 'reordered sensing layout', 'coarse-W', 'Recovered Q (known)', 'Recovered Q (unknown)']
    plot_heat(df_new, target_cols=target_cols, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                    saveFlag=True, showFlag=True, figsize = (3.5*len(target_cols), 2.8*df_new.shape[0]))) # 'Blues'



