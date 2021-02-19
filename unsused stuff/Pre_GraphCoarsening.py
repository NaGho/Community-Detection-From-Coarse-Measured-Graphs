# In the name of God
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from GraphL_Utils import CoarseningParams, fine_mat_gen, MapFineToCoarse, plot_graph_density, SparsityCore, PlotCore
from GraphL_Utils import Fine2CoarseCore, FineMatCore, CoarseMatCore, sparsity_eval, Gdisparity_function
from GraphL_Utils import plot_array, FineCoarseMode, NormalDistProp, DeltaDistProp, SBMDistProp, UniformDistProp
from GraphL_Utils import MixtureDistProp, calc_all_sparsities, calc_Gdisparity, plot_line, plot_scatter
from GraphL_Utils import plot_graphScaling, plot_hist_text, plot_heat_text, graphPruning
from GraphL_Utils import arrayScaling, arrayShifting, arrayMaxForcing, arrayMinForcingScaling
import pandas as pd
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', 1000)

# Test example: 
# sparsity_core = SparsityCore(init=0, end=8, KernelBW=0.1)
# x=np.array([1,2,7])
# S = Gdisparity_function(x, sparsity_core)


## 
# Hyper parameters
V_array = [20] # [20, 50, 100, 200, 500, 1000] # 
N_array = [20] # [200, 75, 25, 15] # [50] # 
KernelBW_array = [0.01]
dist_case_array = [21] # [10, 11, 12, 13] # [14, 15] # np.arange(10) # 
fine_sample_size = 1
reorder_num_cluster = 4

SchCompFlag = False
linearFlag = not SchCompFlag
lin_regularity_array = [1] #np.arange(0, 1, 0.05) #  
overlap_array = [0] # np.arange(0, 1, 0.05) #    
scaling = 'row_normalize'

graphCoarseningStr = 'Schur-Complement' if SchCompFlag else 'Linear'

def select_fine_distribution(size, case, reorder_num_cluster=reorder_num_cluster, inner_case=None):
#         return FineMatCore(size=size, distribution=SBMDistProp(num_communities=num_communities, pre_V=V), laplacian=SchCompFlag)
    scale = 1
    if(case==0):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=0.4*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==1):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=-0.1*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==2):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0*scale, end=0.4*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==3):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0*scale, end=0.8*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==4):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0*scale, sd=0.1*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
    elif(case==5):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0*scale, sd=0.18*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
    elif(case==6):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.5*scale, sd=0.18*scale), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
    elif(case==7):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=2, \
                                            inner_dist_array=[NormalDistProp(-0.5*scale, 0.1*scale), NormalDistProp(0.5*scale, 0.1*scale)], \
                                            dist_coeffs=None), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==8):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0*scale, 0.03*scale), NormalDistProp(0.05*scale, 0.03*scale), \
                                                              NormalDistProp(-0.05*scale, 0.03*scale), NormalDistProp(-0.1*scale, 0.03*scale)], \
                                            dist_coeffs=None), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==9):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0*scale, 0.1*scale), NormalDistProp(0.25*scale, 0.1*scale), \
                                                              NormalDistProp(-0.5*scale, 0.1*scale), NormalDistProp(0.5*scale, 0.1*scale)], \
                                            dist_coeffs=None), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    
#     -----------------------------------------------------------------------------------------------------------------
## sample_hist_comp_sparsities
    elif(case==10):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==11):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==12):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=0.1), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
    elif(case==13):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(-0.6, 0.1), NormalDistProp(-0.2, 0.1), \
                                                              NormalDistProp(0.2, 0.1), NormalDistProp(0.6, 0.1)], \
                                            dist_coeffs=None), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
#     -----------------------------------------------------------------------------------------------------------------
    elif(case==14):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=5, sd=4), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
    elif(case==15):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=10, sd=4), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)  
#     -----------------------------------------------------------------------------------------------------------------
    elif(case==16):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag) 
    elif(case==17):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0, 0.5), NormalDistProp(1, 0.5), \
                                                              NormalDistProp(-1, 0.5), NormalDistProp(2, 0.5)], \
                                            dist_coeffs=None), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==18):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=1), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
#     -----------------------------------------------------------------------------------------------------------------
## intro_Gdisparity_motivation
    elif(case==19):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.25, sd=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
    elif(case==20):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=2, sd=0.15), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
#     -----------------------------------------------------------------------------------------------------------------
## Pruning fine compare
    elif(case==21):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=1, sd=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
#     -----------------------------------------------------------------------------------------------------------------
## Pruning coarse distinguish
    elif(case==24):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.5, sd=0.5), reorder_num_cluster=reorder_num_cluster, laplacian=SchCompFlag)
#     -----------------------------------------------------------------------------------------------------------------


## Pruning fine compare
if(True):
    fineMatFunction_array = [arrayShifting(0), arrayShifting(0.2), arrayMinForcingScaling(0.5,0.7)]    # , arrayShifting(0.3)
else:
    fineMatFunction_array = [arrayShifting(0)]
    
distributionStr = [ 'Delta/shift=0.4', 'Delta/shift=-0.1', 'Uniform[0,0.4]', 'Uniform[0,0.8]', 'Gaussian(mean=0, std=0.1)',\
                    'Gaussian(mean=0, std=0.18)','Gaussian(mean=0.5, std=0.18)', 'Mix 2 Gaussians\nmeans=[-0.5,0.5], std=0.1', \
                    'Mix 4 Gaussians\nmeans=[-0.1,-0.05,0,0.05], std=0.03', 'Mix 4 Gaussians\nmeans=[-0.5,0,0.25,0.5], std=0.1',\
                    '','','','','','','Gaussian(0,1)','Mix 4 Gaussians\nmeans=[-1,0,1,2], std=0.5','Uniform[0,1]', 'Gaussian', 'Gaussian',\
                    '','','']
    
#  
df = pd.DataFrame(columns=['fine-size','fine-W', 'coarse-size','coarse-W', 'Fine-graph-distribution','overlap'])
df[['fine-size', 'coarse-size']] = df[['fine-size', 'coarse-size']].astype('int') 
# int32, float64, int32, float64
#'int32', 'float64', 'int32', 'float64'
# np.int64, np.float64, np.int64, np.float64
for V in V_array:
  for dist_case in dist_case_array:
    fine_core = select_fine_distribution(size=V, reorder_num_cluster=reorder_num_cluster, case=dist_case)
    if(fine_core is None):
        raise ValueError('fine-core is None')
    for fine_sample_counter in np.arange(fine_sample_size):
        W_init = fine_mat_gen(fine_core) #W.min() W.max()
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
                    for KernelBW in KernelBW_array:   
                        coarse_core = CoarseMatCore(size=N)
                        fine_coarse_mode = FineCoarseMode(Schur_comp=SchCompFlag, linear=linearFlag, overlap=overlap, \
                                                          lin_regularity=lin_regularity, scaling=scaling) 
                        fine2Coarse_core = Fine2CoarseCore(N, V, mode=fine_coarse_mode, subsample_set=subsample_set)
                        param = CoarseningParams(fine_core=fine_core, coarse_core=coarse_core, \
                                                 fine2Coarse_core=fine2Coarse_core, sparsity_core=None)
                        W_tilde = MapFineToCoarse(W, param)
                        df_in = pd.DataFrame({'fine-size':[V], 'fine-W':[W],'coarse-size':[N],'coarse-W':[W_tilde], \
                                              'Fine-graph-distribution':[distributionStr[dist_case]], 'overlap': overlap}) 
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

vec_all = np.concatenate((np.concatenate([np.reshape(W, -1) for W in df['fine-W'].values])\
                        , np.concatenate([np.reshape(W, -1) for W in df['coarse-W'].values]))) 

# eval_df = pd.DataFrame()
# eval_df['Fine-graph-distribution'] = df['Fine-graph-distribution']
# eval_df['mins'] = [W.min() for W in df['coarse-W'].values]
# eval_df['maxs'] = [W.max() for W in df['coarse-W'].values]

#TODO make if more efficient, not concatenating all  just to get min and max
# vec_all.shape    np.array(np.concatenate(W_tilde_arr, axis=2)).shape   W.shape
sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=np.max(df['fine-size'].values))
# sparsity_coarse_array = [Gdisparity_function(W_tilde, sparsity_core) for W_tilde in W_tilde_arr] 
sparsity_fine = Gdisparity_function(W, sparsity_core)
sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='coarse-W', ColExtStr='coarse-', sparsity_core=sparsity_core, cloningInvFlag = True)
sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='fine-W', ColExtStr='fine-', sparsity_core=sparsity_core, cloningInvFlag = True)

if(False):
    groupby_cols = ['Fine-graph-distribution']
    x_col = 'overlap' # 'coarse-size' # 
    target_cols = ['coarse-Gdisparity','fine-Gdisparity']
    if(SchCompFlag):
        figName = 'Synthetic_{}_{}_vs_{}_N={}_V={}_kernelBW={}'\
            .format(graphCoarseningStr, '-'.join(target_cols), x_col, N_array[0], V_array[0], KernelBW_array[0])
    else:
        figName = 'Synthetic_{}_vs_{}_N={}_V={}_linearhom={}_overlap={}_kernelBW={}_scaling={}'\
            .format(graphCoarseningStr, '-'.join(target_cols), x_col, N_array[0], V_array[0], lin_regularity_array[0],\
                     overlap_array[0], KernelBW_array[0], scaling)
    plot_line(df, x_col=x_col, target_cols=target_cols, groupby_cols=groupby_cols, plot_core=PlotCore(title='', figName=figName,\
                                                                           saveFlag=False, showFlag=True, log_scale=True,\
                                                                           ylabel='Gdisparity'))
    
if(False):
    figName = 'Sample_graphScaling_{}_{}'.format(graphCoarseningStr, distributionStr[dist_case_array[0]])
    plot_graphScaling(df, plot_core=PlotCore(sparsity_core=sparsity_core, log_scale=False, match_ranges=False,\
                                               num_bins=None, saveFlag=False, figName=figName, figsize = (10, 10), dpi=120))
    
    
if(False):
    x_col = 'fine-Gdisparity' # 'fine-L1Norm' #  'coarse-size' # 
    y_col = 'coarse-Gdisparity' # 'coarse-L1Norm' # 
    hue_col = 'Fine-graph-distribution' # None # 'coarse-size'
    figName = 'Scatter_{}_{}_vs_{}_forDif_{}'.format(graphCoarseningStr, x_col, y_col, hue_col)
    plot_scatter(df, x_col=x_col, y_col=y_col, hue_col=hue_col, style=hue_col,\
                                                         plot_core=PlotCore(title='', figName=figName,\
                                                                        saveFlag=False, showFlag=True, log_scale=True, log_xscale=True))

## sample_hist_comp_sparsities
if(False):
    vec_all = np.concatenate([np.reshape(W, -1) for W in df['fine-W'].values])
    sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
    sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply='fine-W', ColExtStr='fine-', sparsity_core=sparsity_core)
    prestr = 'fine-'
    figName = 'sample_hist_comp_sparsities'.format()
    hist_col = prestr+'W'
    Gdisparity_col = prestr+'Gdisparity'
    if(False):
        df[Gdisparity_col] = df[Gdisparity_col].apply(np.log)
        df.rename(columns={Gdisparity_col: 'log-'+Gdisparity_col})
        Gdisparity_col = 'log-'+Gdisparity_col
        
    text_cols = [Gdisparity_col, prestr+'SparsityL0', prestr+'SparsityL1', prestr+'GiniIndex', prestr+'Hoyer'] #,'fine-minusLoGdisparity','fine-k4Sparsity','fine-l2l1Sparsity']
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, remove_str=prestr, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (6, 8), edgecolor='k'))


if(False):
    prestr = 'fine-'
#     cloningInvFlag = False
#     vec_all = np.concatenate([np.reshape(W, -1) for W in df[prestr+'W'].values])
#     sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
#     sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply=prestr+'W', ColExtStr=prestr, \
#                                             sparsity_core=sparsity_core, cloningInvFlag = cloningInvFlag)
    figName = 'intro_Gdisparity_motivation'.format()
    hist_col = prestr+'W'
    text_cols = [prestr+'SparsityL0',prestr+'SparsityL1',prestr+'GiniIndex',\
                   prestr+'Hoyer',prestr+'minusLoGdisparity', prestr+'k4Sparsity',prestr+'l2l1Sparsity']
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (6, 6), edgecolor='k'))
    
if(False):
    prestr = 'coarse-'
#     cloningInvFlag = False
#     vec_all = np.concatenate([np.reshape(W, -1) for W in df[prestr+'W'].values])
#     sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=5) #  np.max(df['fine-size'].values)
#     sparsity_core, df = calc_all_sparsities(df, KernelBW, columnToApply=prestr+'W', ColExtStr=prestr, \
#                                             sparsity_core=sparsity_core, cloningInvFlag = cloningInvFlag)
    figName = 'intro_Gdisparity_motivation'.format()
    hist_col = prestr+'W'
    text_cols = [prestr+'SparsityL0',prestr+'SparsityL1',prestr+'GiniIndex',\
                   prestr+'Hoyer',prestr+'minusLoGdisparity',prestr+'k4Sparsity',prestr+'l2l1Sparsity']
#     plot_graphScaling(df, hist_col=hist_col,  target_cols=target_cols, heatmap=False, plot_core=PlotCore(sparsity_core=sparsity_core, log_scale=False, match_ranges=False,\
#                                                num_bins=None, saveFlag=False, figName=figName, figsize = (6, 6), dpi=120, edgecolor='k'))
    
    plot_hist_text(df, hist_col=hist_col,  text_cols=text_cols, remove_str=prestr, plot_core=PlotCore(title='', figName=figName, match_ranges=True,\
                                                                        saveFlag=False, showFlag=True, figsize = (18, 8), edgecolor='k',\
                                                                        ColorList=['deepskyblue', 'red', 'limegreen']))
    

## Pruning fine compare
if(True):
    figName = 'intro_pruning'.format()
    prestr = 'fine-'
    target_col = prestr+'W'
    
    cols_str = ['Pruned/fixed threshold', 'Pruned/fixed edgeDensity', 'Pruned/Gdisparity as edgeDensities']
    
    df[cols_str[0]] = df[target_col].apply(graphPruning, args=[0.8, None]) # df[target_col].iloc[0].max()
    
    df[cols_str[1]] = df[target_col].apply(graphPruning, args=[None, 0.2])
    
    Gdisparity_EdgeDensity = df[prestr+'Gdisparity']/df[prestr+'Gdisparity'].sum()
    df[cols_str[2]] = [graphPruning(df[target_col].iloc[i], None, 1-Gdisparity_EdgeDensity.iloc[i]) for i in np.arange(df[target_col].size)]
    
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
           else 'Kernel-Bandwidth' if np.all(xAxisPlot==KernelBW_array)\
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

