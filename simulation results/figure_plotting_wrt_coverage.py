import numpy as np
import matplotlib
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
from Community_utils import SSBMCore, measuringCore, CoarseningCommunityParams, evalCommunityRecovery
from Community_utils import recoverCommunities, GenerateSBMLinCoarsMat, mat2List, min_coverageSBM
from Community_utils import UB_Failure_Error_Function
import pandas as pd

windows = True
Community_Recover = True
ErrorBound_Calc = True
n_array = [30000]
K_array = [5] 
nu_array = [2, 3]
m_array = [400] 

df = pd.read_csv('out_wrt_coverage_for_nu_m400.csv') # out_wrt_coverage_for_nu_m400
    
print(df[['n', 'm', 'r', 'K']].join(df.loc[:, df.columns.str.contains('Recovery Error')]))

value_cols = ['Failure UB Error', 'Failure UB Error (Lowest U)','Failure UB Error (Highest U)']
df = df.melt(id_vars=list(set(df.columns).difference(set(value_cols))), value_vars=value_cols, \
                                    value_name='Failure UB Error', var_name='Low-Mean-High UB Error')


df = df[df.nu==2]
df = df.rename(columns={'Failure UB Error': 'This Paper: Theoretical Error Bound',\
                   'NF1 Recovery Error with M-NMF':'Recovery Error using M-NMF',\
                   'NF1 Recovery Error with SLPA': 'Recovery Error using SLPA',
                   'NF1 Recovery Error with nnsed': 'Recovery Error using NNSED',
                   'NF1 Recovery Error with BigClam': 'Recovery Error using BigClam',
                   'r': 'Coverage Size'})



if(Community_Recover and ErrorBound_Calc):
    x_col = 'Coverage Size' # 'r' # 'n' # 'r' # 'K' # 'nu' # 'tauTilde' # 'alpha'
    y_cols = ['This Paper: Theoretical Error Bound', 'Recovery Error using M-NMF', 'Recovery Error using SLPA', \
                                  'Recovery Error using NNSED', 'Recovery Error using BigClam'] 
    style_col = None # 'nu' # None ,   'p',  'nu'
    hue_col = None # if style_col is not None else 'p' # 'nu'
    y_col = "Error"
    var_name = " "
    df = df.melt(id_vars=[x_col, hue_col] if hue_col is not None else [x_col, style_col] if style_col is not None else [x_col], \
                                         value_vars=y_cols, value_name=y_col, var_name=var_name)

    title = 'synthetic_UB_Errors'
    figName = title + '_wrt_{}_n{}_m{}_K{}_nu{}'.format(x_col, n_array[0], m_array[0], K_array[0], nu_array[0])
    relPlot(df, x_col=x_col, y_col=y_col, hue_col=var_name, style_col=var_name, \
                                plot_core=PlotCore(title='', figName=figName, \
                                    saveFlag=True, showFlag=windows, log_scale=False, aspect=0.5, \
                                        minX_axis=np.min(df[x_col])-2, maxX_axis=np.max(df[x_col])+5,\
                                            minY_axis=np.min(df[y_col])-0.01, maxY_axis=np.max(df[y_col])+0.01, palette_cmap='tab10',\
                                                paper_rc = {'lines.linewidth': 4, 'lines.markersize': 8} ))
    # palettes = ['BuPu', 'hot', sns.color_palette("mako_r", 6) , "tab10", 'RdYlBu' , 'Paired' , 'Dark2', "Reds", sns.cubehelix_palette(8)]
    
    