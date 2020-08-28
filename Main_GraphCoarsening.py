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
import pandas as pd
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)

# Test example: 
# sparsity_core = SparsityCore(init=0, end=8, KernelBW=0.1)
# x=np.array([1,2,7])
# S = Gsparsity_function(x, sparsity_core)


## 
# Hyper parameters
V_array = [500] # [20, 50, 100, 200, 500, 1000] # 
N_array = [50] # list(np.arange(5, 200, 10)) # [200, 75, 25, 15] # [40] #  
KernelBW = 0.01
# Gsparsity of x and y are proportional => [23, 24]
# Homogeneity  => [25, 26]
# Mean => ['Gaussian','Uniform']
# Std => ['Gaussian','Uniform']
dist_case_array = ['Gaussian','Uniform', 'Mixture2Gaussian'] # , 'Mixture3Gaussian', 'Mixture4Gaussian', 'Delta'
# np.arange(10) #  [3, 4, 5] # [21] # [10, 11, 12, 13] # [14, 15] # 
fine_sample_size = 2
reorder_num_cluster = 3
zero_coeffs = [0.1] # list(np.arange(0.1, 1, 0.05)) # [0.1] # 
dist_case_mean_array = [0, 3] # list(np.arange(-2, 2.1, 0.2)) # [0] # 
dist_case_std_array = [1/5] # list(np.arange(1e-4, 2.1, 0.2)) #     


SchCompFlag = False
linearFlag = not SchCompFlag
lin_regularity_array = list(np.arange(0, 1, 0.1)) # [1] # 
overlap_array = [0] # list(np.arange(0, 0.95, 0.05)) #         
scaling = 'row_normalize'

graphCoarseningStr = 'Schur-Complement' if SchCompFlag else 'Linear'

 
## Pruning fine compare
if(False):
    fineMatFunction_array = [arrayShifting(0), arrayShifting(0.2), arrayMinForcingScaling(0.5,0.7)]    # , arrayShifting(0.3)
else:
    fineMatFunction_array = [arrayShifting(0)]
    
distributionStr = ['Delta(shift=0.2), p_0=0.6,', \
                   'Delta(shift=-0.1), p_0=0.3', \
                   'Uniform[-0.5,0.5], p_0=0.2', \
                   'Uniform[0,1], p_0=0.2', \
                   'Gaussian(mean=0, std=1), p_0=0.2',\
                   'Gaussian(mean=-1, std=0.5), p_0=0',\
                   'Gaussian(mean=0.5, std=1), p_0=0.2', \
                   'Mix 2 Gaussians\nmeans=[-0.2,0.8], std=0.1, p_0=0.2', \
                   'Mix 4 Gaussians\nmeans=[-0.1,-0.05,0,0.05], std=0.03, p_0=0.2', \
                   'Mix 4 Gaussians\nmeans=[-0.5,0,0.25,0.5], std=0.1, p_0=0.2',\
                   'Delta','Uniform','Normal','Mixture4',\
                   'Gaussian','Gaussian',\
                   'Gaussian(0,1)', 'Mix 4 Gaussians\nmeans=[-1,0,1,2], std=0.5','Uniform[0,1]',\
                   'Gaussian', 'Gaussian',\
                   'Gaussian',\
                   'Gaussian',\
                   'Uniform[0,1]','Gaussian(mean=0, std=1)',\
                   'Uniform[0,1]','Gaussian(mean=0, std=1)']
    
#  
df = pd.DataFrame(columns=['fine-size','fine-W', 'coarse-size','coarse-W', 'fine-distribution','overlap'])
df[['fine-size', 'coarse-size']] = df[['fine-size', 'coarse-size']].astype('int') 
# int32, float64, int32, float64
#'int32', 'float64', 'int32', 'float64'
# np.int64, np.float64, np.int64, np.float64
for V in V_array:
  for dist_case in dist_case_array:
    for zero_coeff in zero_coeffs:
        for mean in dist_case_mean_array:
            for std in dist_case_std_array:
                fine_core = select_fine_distribution(size=V , case=dist_case, zero_coeff=zero_coeff, mean=mean, sd=std)
                if(fine_core is None):
                    raise ValueError('fine-core is None')
                for fine_sample_counter in np.arange(fine_sample_size):
                    W_init = fine_mat_gen(fine_core) # W.min() W.max()
                    for fineMatFunction in fineMatFunction_array:
                        W = fineMatFunction.apply(W_init)
            #             if(True):
            #                 W[np.argwhere(np.abs(W)<=2e-5)] = 0
                        for N in N_array:
                            num_communities = N
                            community_size = int(V/num_communities)
                            subsample_set = np.arange(0, V, community_size) # np.random.randint(0,V,size=(N,)) # np.arange(0, V, community_size) #np.arange(20)
                            for lin_regularity in lin_regularity_array:
                              for overlap in  overlap_array:   
                                coarse_core = CoarseMatCore(size=N)
                                fine_coarse_mode = FineCoarseMode(Schur_comp=SchCompFlag, linear=linearFlag, overlap=overlap, \
                                                                  lin_regularity=lin_regularity, scaling=scaling) 
                                fine2Coarse_core = Fine2CoarseCore(N, V, mode=fine_coarse_mode, subsample_set=subsample_set)
                                param = CoarseningParams(fine_core=fine_core, coarse_core=coarse_core, \
                                                         fine2Coarse_core=fine2Coarse_core, sparsity_core=None)
                                W_tilde = MapFineToCoarse(W, param)
                                df_in = pd.DataFrame({'fine-size':[V], 'fine-W':[W],'coarse-size':[N],'coarse-W':[W_tilde], \
                                                      'fine-distribution': [(dist_case+' mean={0:.1f}, std={1:.1f}').format(mean,std)] \
                                                      if isinstance(dist_case, str) and False\
                                                      else [(dist_case+' mean={0:.1f}').format(mean)] if isinstance(dist_case, str) and True\
                                                      else [dist_case] if isinstance(dist_case, str) \
                                                      else [distributionStr[dist_case]], \
                                                      'mean':[mean], 'std':[std], 'p0': [zero_coeff], \
                                                      'overlap': [overlap], 'linear-regularity':[lin_regularity]}) 
                                #TODO make if more efficient, not saving same W over and over
                                df = df.append(df_in) 
                                 
                                if(False):
                                    vec_all = np.concatenate((np.reshape(W, -1), np.reshape(np.array(W_tilde), -1)))
                                    param.sparsity_core = sparsity_core = \
                                                        SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=V)
                                    if(SchCompFlag):
                                        figName = 'Sparsening_{}'.format(graphCoarseningStr)
                                    else:
                                        figName = 'Sparsening_{}_Linear={}_overlap={}_scaling={}'.format\
                                                            (graphCoarseningStr, lin_regularity, overlap, scaling)
                                    plot_graph_density(W_tilde, W, plot_core=\
                                                       PlotCore(sparsity_core=sparsity_core, log_scale=False, match_ranges=False,\
                                                           num_bins=None, saveFlag=False, \
                                                           figName=figName))

vec_all = np.concatenate((np.concatenate([np.reshape(W, -1) for W in df['fine-W'].values]), \
                          np.concatenate([np.reshape(W, -1) for W in df['coarse-W'].values]))) 

# eval_df = pd.DataFrame()
# eval_df['fine-distribution'] = df['fine-distribution']
# eval_df['mins'] = [W.min() for W in df['coarse-W'].values]
# eval_df['maxs'] = [W.max() for W in df['coarse-W'].values]

#TODO make if more efficient, not concatenating all  just to get min and max
# vec_all.shape    np.array(np.concatenate(W_tilde_arr, axis=2)).shape   W.shape
sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=np.max(df['fine-size'].values))
# sparsity_coarse_array = [Gsparsity_function(W_tilde) for W_tilde in W_tilde_arr] 
sparsity_fine = Gsparsity_function(W)
sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='coarse-W', ColExtStr='coarse-', sparsity_core=sparsity_core, cloningInvFlag = True)
sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='fine-W', ColExtStr='fine-', sparsity_core=sparsity_core, cloningInvFlag = True)

## sample_hist_comp_sparsities
if(df.shape[0]<6):
    vec_all = np.concatenate([np.reshape(W, -1) for W in df['fine-W'].values])
    sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
    sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='fine-W', ColExtStr='fine-', sparsity_core=sparsity_core)
    prestr = 'fine-'
    figName = 'sample_hist_comp_sparsities'.format()
    hist_col = prestr+'W'
    Gsparsity_col = prestr+'Gsparsity'
    if(False):
        df[Gsparsity_col] = df[Gsparsity_col].apply(np.log)
        df.rename(columns={Gsparsity_col: 'log-'+Gsparsity_col})
        Gsparsity_col = 'log-'+Gsparsity_col
        
    text_cols = [Gsparsity_col, prestr+'SparsityL0', prestr+'SparsityL1', prestr+'GiniIndex', prestr+'Hoyer'] #,'fine-minusLoGsparsity','fine-k4Sparsity','fine-l2l1Sparsity']
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, remove_str=prestr, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (6, 8), edgecolor='k'))



if(False):
    x_col = 'fine-Gsparsity'
    y_col = 'coarse-Gsparsity'
    plot_single_regression(df, x_col=x_col, y_col=y_col, hue_col='fine-distribution',\
                                plot_core=PlotCore(title='', figName='Gsparsity_fine_Coarse_regression_{}'.format(graphCoarseningStr),\
                                   saveFlag=False, showFlag=True, log_scale=False))
    
if(True):
    minLenList = 3
    groupby_col = 'fine-distribution'
    if(len(lin_regularity_array)>minLenList):
        x_col = 'linear-regularity' 
    elif(len(N_array)>minLenList):
        x_col = 'coarse-size' 
    elif(len(overlap_array)>minLenList):
        x_col = 'overlap' 
    elif(len(dist_case_mean_array)>minLenList):
        x_col = 'mean'
    elif(len(dist_case_std_array)>minLenList):
        x_col = 'std'
    elif(len(zero_coeffs)>minLenList):
        x_col = 'p0'
    else:
        x_col = 'fine-Gsparsity'
#     if(True):
#         groupby_col = 'full-fine-distribution'
#         df = df.melt(id_vars=[x_col,groupby_col], value_vars= ['mean','std'], value_name=groupby_col, var_name='whatever')
    target_cols = ['coarse-Gsparsity','fine-Gsparsity']
    if(SchCompFlag):
        figName = 'Gsparsity_wrt_{}_{}_N={}_V={}_mean={}_std={}'\
            .format(x_col, graphCoarseningStr, N_array[0], V_array[0], dist_case_mean_array[0], dist_case_std_array[0])
    else:
        figName = 'Gsparsity_wrt_{}_{}_N={}_V={}_mean={}_std={}_linearhom={}_overlap={}_scaling={}'\
            .format(x_col, graphCoarseningStr, N_array[0], V_array[0], dist_case_mean_array[0], dist_case_std_array[0], \
                    lin_regularity_array[0], overlap_array[0], scaling)
    plot_line(df, x_col=x_col, target_cols=target_cols, groupby_col=groupby_col, plot_core=PlotCore(title='', figName=figName,\
                                                                           saveFlag=True, showFlag=True, log_scale=False,\
                                                                           ylabel='Gsparsity'))
    
if(False):
    figName = 'Sample_graphScaling_{}_{}'.format(graphCoarseningStr, distributionStr[dist_case_array[0]])
    plot_graphScaling(df, plot_core=PlotCore(sparsity_core=sparsity_core, log_scale=False, match_ranges=False,\
                                               num_bins=None, saveFlag=False, figName=figName, figsize = (10, 10), dpi=120))
    
    
# Fine-Coarse Gsparsity Scatter Plot    
if(False):
    x_col = 'fine-Gsparsity' # 'fine-L1Norm' #  'coarse-size' # 
    y_col = 'coarse-Gsparsity' # 'coarse-L1Norm' # 
    hue_col = 'fine-distribution' # None # 'coarse-size'
    figName = 'Scatter_{}_{}_vs_{}_forDif_{}'.format(graphCoarseningStr, x_col, y_col, hue_col)
    plot_scatter(df, x_col=x_col, y_col=y_col, hue_col=hue_col, style=hue_col,\
                                                         plot_core=PlotCore(title='', figName=figName,\
                                                                saveFlag=True, showFlag=True, log_scale=False, log_xscale=False))


if(False):
    prestr = 'fine-'
#     cloningInvFlag = False
#     vec_all = np.concatenate([np.reshape(W, -1) for W in df[prestr+'W'].values])
#     sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
#     sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply=prestr+'W', ColExtStr=prestr, \
#                                             sparsity_core=sparsity_core, cloningInvFlag = cloningInvFlag)
    figName = 'intro_Gsparsity_motivation'.format()
    hist_col = prestr+'W'
    text_cols = [prestr+'SparsityL0',prestr+'SparsityL1',prestr+'GiniIndex',\
                   prestr+'Hoyer',prestr+'minusLoGsparsity', prestr+'k4Sparsity',prestr+'l2l1Sparsity']
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (6, 6), edgecolor='k'))
    
if(False):
    prestr = 'coarse-'
#     cloningInvFlag = False
#     vec_all = np.concatenate([np.reshape(W, -1) for W in df[prestr+'W'].values])
#     sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
#     sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply=prestr+'W', ColExtStr=prestr, \
#                                             sparsity_core=sparsity_core, cloningInvFlag = cloningInvFlag)
    figName = 'intro_Gsparsity_motivation'.format()
    hist_col = prestr+'W'
    text_cols = [prestr+'SparsityL0',prestr+'SparsityL1',prestr+'GiniIndex',\
                   prestr+'Hoyer',prestr+'minusLoGsparsity',prestr+'k4Sparsity',prestr+'l2l1Sparsity']
#     plot_graphScaling(df, hist_col=hist_col,  target_cols=target_cols, heatmap=False, plot_core=PlotCore(sparsity_core=sparsity_core, log_scale=False, match_ranges=False,\
#                                                num_bins=None, saveFlag=False, figName=figName, figsize = (6, 6), dpi=120, edgecolor='k'))
    
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, remove_str=prestr, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (18, 8), edgecolor='k',\
                                                                        ColorList=['deepskyblue', 'red', 'limegreen']))
    

## Pruning fine compare
if(False):
    figName = 'intro_pruning'.format()
    prestr = 'fine-'
    target_col = prestr+'W'
    
    cols_str = ['Pruned/fixed threshold', 'Pruned/fixed edgeDensity', 'Pruned/Gsparsity as edgeDensities']
    
    df[cols_str[0]] = df[target_col].apply(graphPruning, args=[0.8, None]) # df[target_col].iloc[0].max()
    
    df[cols_str[1]] = df[target_col].apply(graphPruning, args=[None, 0.2])
    
#     Gsparsity_EdgeDensity = df[prestr+'Gsparsity']/df[prestr+'Gsparsity'].sum()
    df[cols_str[2]] = [graphPruning(df[target_col].iloc[i], None, 1-df[prestr+'Gsparsity'].iloc[i]) for i in np.arange(df[target_col].size)]
    
    for col_str in cols_str:
        _, df = calc_all_sparsities(df, KernelBW, columnToApply=col_str, ColExtStr=col_str+'-', sparsity_core=sparsity_core)
    
    
    plot_heat_text(df, target_cols=[prestr+'W', cols_str[0], cols_str[1], cols_str[2]],\
                        text_cols=[[prestr+'SparsityL0',prestr+'SparsityL1',prestr+'GiniIndex',prestr+'Hoyer'],\
#                                    [col1str+'-'+'SparsityL0',col1str+'-'+'-'+'GiniIndex',col1str+'-'+'Hoyer'],\
#                                    [col2str+'-'+'SparsityL0',col2str+'-'+'GiniIndex',col2str+'-'+'Hoyer'],\
#                                    [col3str+'-'+'SparsityL0',col3str+'-'+'GiniIndex',col3str+'-'+'Hoyer']\
                                   [],[],[]],\
                             remove_str=prestr, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                    saveFlag=False, showFlag=True, figsize = (12, 7), palette_cmap=None)) # 'Blues'




if(False):  
    plot_regression(df, groupby_col=groupby_col, target_cols=target_cols, \
                                plot_core=PlotCore(title='', figName='Synthetic_{}_vs_{}_regression_'.\
                                                   format('-'.join(target_cols), groupby_col),\
                                   saveFlag=False, showFlag=True, log_scale=False))
     
    xAxisPlot = overlap_array
    xlabel =    'N (coarse-graph size)' if np.all(xAxisPlot==N_array)\
           else 'Linear-regularity' if np.all(xAxisPlot==lin_regularity_array)\
           else 'Overlap-extent' if np.all(xAxisPlot==overlap_array)\
           else 'Unknown'
    plot_array(t=xAxisPlot, arrays=[sparsity_fine*np.ones_like(sparsity_coarse_array), sparsity_coarse_array], \
                            plot_core= PlotCore(legends=['Fine graph', 'Coarse graph'], \
                                   sparsity_core=sparsity_core, \
                                   xlabel=xlabel, ylabel='Sparsity', title='', \
                                            figName=figName,\
                                                   saveFlag=False))
             

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


END = 1

