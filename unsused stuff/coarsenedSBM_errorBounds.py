import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import sys
import seaborn as sns
from GraphL_Utils import CoarseningParams, fine_mat_gen, MapFineToCoarse, plot_graph_density, SparsityCore, PlotCore
from GraphL_Utils import Fine2CoarseCore, FineMatCore, CoarseMatCore, sparsity_eval, Gsparsity_function
from GraphL_Utils import plot_array, FineCoarseMode, NormalDistProp, DeltaDistProp, SBMDistProp, UniformDistProp
from GraphL_Utils import MixtureDistProp, calc_all_sparsities, plot_line, plot_scatter, plot_regression
from GraphL_Utils import plot_graphScaling, plot_hist_text, plot_heat_text, graphPruning, plot_single_regression
from GraphL_Utils import arrayScaling, arrayShifting, arrayMaxForcing, arrayMinForcingScaling, select_fine_distribution
from GraphL_Utils import df_empty
import pandas as pd
from math import *
import itertools
from scipy.stats import norm
from Community_utils import UB_ML_Failure_Error_Function, nCr, profile2community, CH_divergence, generateNormalizedProfileSet
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)

    
    
n_array = [1e4] # np.arange(2.5e5, 1e6, 1e5) # [1e5, 5e5, 1e6] # [20, 50, 100, 200, 500, 1000] # [1000, 5000,  10000] # 
K_array = [5] # [5, 10, 20] #  
# as much as we want L to be large, we want it to be as much smaller and further from \sqrt(nlog(n)) as possible
L_array = [10, 50, 100, 200, 500, 1e3] #  [5, 10, 20, 40, 60, 100] # [5, 10, 20, 40, 60, 100, 200, 1000, 5000] # 


alpha_array = [100] # [1.5] # np.arange(1, 52, 8) # 
beta = 10 # 100
## alpha or beta * log(n)/n should be less than 1
df = df_empty(['n','L', 'K', 'C', 'ML Failure UB Log-Error'], ['int', 'int', 'int', 'int', 'float'])
nu_array = [2] # [1, 2] # 

for n in n_array:
    for K in K_array:
#         L_array = list(np.ceil(np.arange(K_array[0], n_array[0]/4, int(np.floor((n_array[0]/4-K_array[0])/3)))/K).astype(int)*K) # list(np.floor(np.arange(180, 420)/K).astype(int)*K) # [50] # [200, 75, 25, 15] # [40] #  [n] #
#         L_array = np.unique(L_array)
        for nu in nu_array:
            for L in L_array:
                C_array =  [10] # [int(n/(2*L))] #  np.arange(10, int(n/L), 5) # [int(n/(3*L))] # [10, 25, 50, 75, 100] # [int(n/L)] # [int(n/(10*L))]
                for alpha in alpha_array:
                    for C in C_array:
                        print('C={}, p={}, q={} '.format(C, df_in['p'], df_in['q']))
                        if(C*L>n):
                            print('C*L>n error!')
                            continue
                        alpha_beta_Gap = (alpha+beta)/2-np.sqrt(alpha*beta)
                        df_in = {'n':n, 'L':L, 'K':K, 'C':C, 'alpha':alpha, 'beta':beta, 'alpha-beta Gap': alpha_beta_Gap, 'nu':nu}
                        UB_ML_Failure_LogError, df_in = UB_Failure_Error_Function(df_in)
                        print('L={}, log-error={}'.format(L, UB_ML_Failure_LogError))
                        df = df.append(df_in, ignore_index=True) 
            
            
# print(df['ML Failure UB Log-Error'])
if(True):
    x_col = 'L' # 'n' # 'C' # 
    target_cols = ['ML Failure UB Log-Error'] # 'ML Failure UB Log-Error',  'alpha-beta Gap'
    hue_col = 'nu' # 'alpha-beta Gap' # 'K' # 'n' # 
    figName = 'UB_ML_Failure_Error_vs_{0:s}_forVarious_{1:s}_gap={2:0.0f}'.format(x_col, hue_col, alpha_beta_Gap)
    plot_line(df, x_col=x_col, target_cols=target_cols, hue_col=hue_col, plot_core=PlotCore(title='', figName=figName,\
                                                                            minY_axis=None, maxY_axis=None, \
                                                                            saveFlag=False, showFlag=True, log_scale=False, \
                                                                            ylabel='ML Failure UB Log-Error'))
    
    
    
    
    
    
    
    
    