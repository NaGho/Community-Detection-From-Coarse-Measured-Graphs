import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import seaborn as sns
import pandas
import random
# from supervised_tasks import LoadCore, data_load, PreProcessing, Task
from scipy.signal import butter, lfilter, freqz
from itertools import islice
# import sys
from dask.dataframe.tests.test_rolling import idx
import pandas as pd
from scipy import stats, signal
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.datasets import make_checkerboard
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import networkx as nx
import snap
from itertools import chain 
from collections import Counter
from scipy import optimize
from sklearn.cluster import KMeans
from numpy.f2py.auxfuncs import isarray
from datetime import datetime
import itertools
from collections import Counter
# from graph_tool import spectral
# from numba import jit, cuda
# import graph_tool.all as gt

np.set_printoptions(precision=2)
RColorBrewer_palette = ['Set1', 'Set2', 'Set3', 'Paired', 'BuPu', 'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral']
ColorList = ['darkred','red', 'orangered', 'darkorange', 'orange']# ['blue', 'deepskyblue','red', 'darkorange', 'gold','limegreen']
colors = ["#c03d56", "#005589","#2d7858"]
customPalette1 = "Paired"# sns.set_palette(sns.color_palette(colors))

def select_fine_distribution(size, case, zero_coeff=None, mean=0, sd=None, min=None, max=None):
#         return FineMatCore(size=size, distribution=SBMDistProp(num_communities=num_communities, pre_V=V), laplacian=SchCompFlag)
    if(case==0):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=0.2 , zero_coeff=0.6) )
    elif(case==1):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=-0.1 , zero_coeff=0.3) )
    elif(case==2):
        return FineMatCore(size=size, distribution=UniformDistProp(start=-0.5 , end=0.5 , zero_coeff=zero_coeff) )
# Fine-Coarse Gsparsity Scatter Plot
    elif(case==3):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0 , end=1 , zero_coeff=zero_coeff) )
    elif(case==4):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0 , sd=1 , zero_coeff=zero_coeff) )  
    elif(case==5):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=-1 , sd=0.5 , zero_coeff=0) )  
    elif(case==6):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.5 , sd=1 , zero_coeff=zero_coeff) )  
    elif(case==7):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=2, \
                                            inner_dist_array=[NormalDistProp(-0.2 , 0.1 ), NormalDistProp(0.8 , 0.1 )], \
                                            dist_coeffs=None, zero_coeff=zero_coeff) )
    elif(case==8):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0 , 0.03 ), NormalDistProp(0.05 , 0.03 ), \
                                                              NormalDistProp(-0.05 , 0.03 ), NormalDistProp(-0.1 , 0.03 )], \
                                            dist_coeffs=None, zero_coeff=zero_coeff) )
    elif(case==9):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0 , 0.1 ), NormalDistProp(0.25 , 0.1 ), \
                                                              NormalDistProp(-0.5 , 0.1 ), NormalDistProp(0.5 , 0.1 )], \
                                            dist_coeffs=None, zero_coeff=zero_coeff) )
    
#     -----------------------------------------------------------------------------------------------------------------
## sample_hist_comp_sparsities
    elif(case==10):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=0.5) )
    elif(case==11):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=0.5) )
    elif(case==12):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=0.1) )  
    elif(case==13):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(-0.6, 0.1), NormalDistProp(-0.2, 0.1), \
                                                              NormalDistProp(0.2, 0.1), NormalDistProp(0.6, 0.1)], \
                                            dist_coeffs=None) )
#     -----------------------------------------------------------------------------------------------------------------
    elif(case==14):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=5, sd=4) )  
    elif(case==15):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=10, sd=4) )  
#     -----------------------------------------------------------------------------------------------------------------
    elif(case==16):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=0.5) ) 
    elif(case==17):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                            inner_dist_array=[NormalDistProp(0, 0.5), NormalDistProp(1, 0.5), \
                                                              NormalDistProp(-1, 0.5), NormalDistProp(2, 0.5)], \
                                            dist_coeffs=None) )
    elif(case==18):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=1) )
#     -----------------------------------------------------------------------------------------------------------------
## intro_Gsparsity_motivation
    elif(case==19):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.25, sd=0.5) )
    elif(case==20):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=2, sd=0.15) )
#     -----------------------------------------------------------------------------------------------------------------
## Pruning fine compare
    elif(case==21):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=1, sd=0.5) )
#     -----------------------------------------------------------------------------------------------------------------
## Pruning coarse distinguish
    elif(case==22):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0.5, sd=0.5) )
#     -----------------------------------------------------------------------------------------------------------------
## Gsparsity of x and y are proportional 
    elif(case==23):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=1, zero_coeff=zero_coeff) )
    elif(case==24):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=1, zero_coeff=zero_coeff) )

## Homogeneity  
    elif(case==25):
        return FineMatCore(size=size, distribution=UniformDistProp(start=0, end=1, zero_coeff=zero_coeff) )
    elif(case==26):
        return FineMatCore(size=size, distribution=NormalDistProp(mean=0, sd=1, zero_coeff=zero_coeff) )
# Parametric     
    numStd = 4
    if(case=='Gaussian'):
        if(sd is None):
            sd = np.max((max-mean, mean-min))/numStd
        return FineMatCore(size=size, distribution=NormalDistProp(mean=mean, sd=sd, zero_coeff=zero_coeff) )
    elif(case=='Uniform'): # (a+b)/2=mean  # (b-a)^2/12=sd 
        if(min is None and max is None):
            max = (2*mean+np.sqrt(12*sd))/2
            min = (2*mean-np.sqrt(12*sd))/2
        elif(min is None):
            min = 2*mean - max
        elif(max is None):
            max = 2*mean - min
        return FineMatCore(size=size, distribution=UniformDistProp(start=min, end=max, zero_coeff=zero_coeff) )
    elif(case=='Mixture2Gaussian'):
        if(min is None and max is None):
            min = 0
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=2, \
                                        inner_dist_array=[NormalDistProp(-1+mean, sd), NormalDistProp(1+mean, sd)], \
                                        dist_coeffs=None, zero_coeff=zero_coeff) ) 
    elif(case=='Mixture3Gaussian'):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=3, \
                                        inner_dist_array=[NormalDistProp(mean, sd), NormalDistProp(1+mean, sd), \
                                                          NormalDistProp(-1+mean, sd)], \
                                        dist_coeffs=None, zero_coeff=zero_coeff) )  
    elif(case=='Mixture4Gaussian'):
        return FineMatCore(size=size, distribution=MixtureDistProp(num_components=4, \
                                        inner_dist_array=[NormalDistProp(-1.5 + mean, sd), NormalDistProp(-0.5 + mean, sd), \
                                                          NormalDistProp(0.5 + mean, sd), NormalDistProp(1.5 + mean, sd)], \
                                        dist_coeffs=None, zero_coeff=zero_coeff) )    
    elif(case=='Delta'):
        return FineMatCore(size=size, distribution=DeltaDistProp(shift=mean , zero_coeff=zero_coeff) ) 
    
    
    

class arrayScaling(object):
    def __init__(self, scale=1):
        self.scale = scale
    def apply(self, arr):   
        return arr*self.scale

class arrayShifting(object):
    def __init__(self, shift=0):
        self.shift = shift
    def apply(self, arr):   
        return arr + self.shift

class arrayMaxForcing(object):
    def __init__(self, percentage=0):
        self.percentage = percentage
    def apply(self, arr):
        x = arr.flatten()   
        sorted = np.sort(x)
        th = sorted[int(np.floor(self.percentage*x.size))]
        arr[arr>th] = th
        return arr
    

class arrayMinForcingScaling(object):
    def __init__(self, forcing_percentage=0, scale=1):
        self.forcing_percentage = forcing_percentage
        self.scale = scale
    def apply(self, arr):
        x = np.abs(arr.flatten()) 
        sorted = np.sort(x)
        th = sorted[int(np.floor(self.forcing_percentage*x.size))]
        arr[np.abs(arr)<th] = 0.01
        return arr*self.scale



class FineCoarseMode():
    def __init__(self, Schur_comp, linear, overlap=0, lin_regularity=1, scaling=1):
        self.Schur_comp = Schur_comp
        self.linear = linear
        self.overlap = overlap # 0: completely disjointed , 1:all the same, full overlap
        self.lin_regularity = lin_regularity # lin_regularity=1.0 : completely homogeneous
        self.scaling = scaling # a number from 0 to 1 - or 'row_normalize' , 'mat_normalize'
        

class CoarseningParams():
    def __init__(self, fine_core=None, coarse_core=None, fine2Coarse_core=None, sparsity_core=None):
        self.fine_core = fine_core
        self.coarse_core = coarse_core
        self.fine2Coarse_core = fine2Coarse_core
        self.sparsity_core = sparsity_core
            
                
class Fine2CoarseCore(object):
    def __init__(self, N, V, mode=None, subsample_set=None, lin_regularity=None, linear_Coarsening_mat=None):
        self.mode = mode
        self.subsample_set = subsample_set
        if(N is not None and subsample_set is None):
                self.subsample_set = np.arange(N)
#        **2 
        self.linear_Coarsening_mat = linCoarseningMatGen(V, N, mode)\
                                            if (mode.linear and linear_Coarsening_mat is None)\
                                                else linear_Coarsening_mat 




class NormalDistProp(object):
    def __init__(self, mean=0, sd=1, zero_coeff=0):
        self.sd = sd
        self.mean = mean
        self.zero_coeff = zero_coeff
        
    def sample(self, V=1):
        size = V**2
        num_zeros = int(np.floor(size*self.zero_coeff))
        W = np.concatenate((np.random.normal(size=(size-num_zeros,)) * self.sd + self.mean, np.zeros((num_zeros,))))
        return np.reshape(W, [V,V])
        
class DeltaDistProp(object):
    def __init__(self, shift=1, zero_coeff=0):
        self.shift = shift  
        self.zero_coeff = zero_coeff
        
    def sample(self, V=1):
        size = V**2
        num_zeros = int(np.floor(size*self.zero_coeff))
        W = np.concatenate((np.ones((size-num_zeros,)) * self.shift, np.zeros((num_zeros,))))
        return np.reshape(W, [V,V])
       
        
class UniformDistProp(object):
    def __init__(self, start=0, end=1, zero_coeff=0):
        self.start = start
        self.end = end
        self.zero_coeff = zero_coeff
        
    def sample(self, V=1): 
        size = V**2
        num_zeros = int(np.floor(size*self.zero_coeff))  
        W = np.concatenate((np.random.uniform(low=self.start, high=self.end, size=(size-num_zeros,)), np.zeros((num_zeros,))))
        return np.reshape(W, [V,V])
        
        

class LogNormalDistProp(object):
    def __init__(self, mean=0, std=1, zero_coeff=0):
        self.mean = mean
        self.std = std
        self.zero_coeff = zero_coeff
        
    def sample(self, size=1): 
        size = V**2
        num_zeros = int(np.floor(size*self.zero_coeff))  
        W = np.concatenate((np.random.lognormal(mean=self.mean, sigma=self.std, size=(size-num_zeros,)), np.zeros((num_zeros,))))
        return np.reshape(W, [V,V])




# class DistProp(object):
#     def __init__(self, mean=0, std=1, zero_coeff=0):
#         self.mean = mean
#         self.std = std
#         self.zero_coeff = zero_coeff
#         
#     def sample(self, size=1): 
#         num_zeros = int(np.floor(size*self.zero_coeff))  
#         return np.concatenate((np.random.(, size=(size-num_zeros,)), np.zeros((num_zeros,))))




   
class FineMatCore(object):
    def __init__(self, size=None, graphGenMode=None, laplacian=False): #normal_dist=NormalDistProp(), delta_dist=DeltaDistProp(),SBM_dist=None, unif_dist=UniformDistProp()):
        self.size = size
        self.graphGenMode = graphGenMode
        self.laplacian = laplacian
#         self.normal_dist = normal_dist
#         self.delta_dist = delta_dist
#         self.SBM_dist = SBM_dist
#         self.unif_dist = unif_dist
      
        

class MixtureDistProp(object):
    def __init__(self, num_components=2, inner_dist_array=[NormalDistProp(-1, 1), NormalDistProp(1, 1)], dist_coeffs=None, zero_coeff=0):
        self.num_components = num_components 
        self.inner_dist_array = inner_dist_array
        if(dist_coeffs is None):
            dist_coeffs = np.ones((num_components,))/num_components
        self.dist_coeffs = dist_coeffs
        self.zero_coeff = zero_coeff
        
    def sample(self, V=1):  
        size = V**2 
        num_zeros = int(np.floor(size*self.zero_coeff))
        unifs = np.random.uniform(size=(size-num_zeros,))
#         unifs = np.reshape(unifs, -1) # unifs.shape
        component_prob = np.cumsum(self.dist_coeffs)
        W = np.ones_like(unifs) * -10000 #W.size
        for i in np.arange(unifs.size): #TODO make this faster
            for j in np.arange(component_prob.size):
                if(unifs[i]<=component_prob[j]):
                    W[i] = self.inner_dist_array[j].sample()
                    break
        W = np.concatenate((W, np.zeros((num_zeros,)))) # np.reshape(W, size)
        return np.reshape(W, [V,V])
        
        
class CoarseMatCore(object):
    def __init__(self, size):
        self.size = size
    
    
class PlotCore(object):
    def __init__(self, sparsity_core=None, log_scale=False, log_xscale=False , match_ranges=False, num_bins=None, \
                 minX_axis=None, maxX_axis=None, minY_axis=None, maxY_axis=None, saveFlag=False, figName=' ', \
                 xlabel=None, ylabel=None, title=None, paper_rc=None, aspect=None,\
                 legends=None, showFlag=True, figsize = (10, 8), dpi=80, fineGraphFlag = None, facecolor=None,\
                 edgecolor=None, ColorList=None, palette_cmap=None, reorder_num_cluster=4, heatmap_reorder=False):
        self.sparsity_core = sparsity_core
        self.log_scale = log_scale
        self.log_xscale = log_xscale
        self.match_ranges = match_ranges
        self.num_bins = num_bins
        self.fineGraphFlag = fineGraphFlag
        self.minX_axis = minX_axis
        self.maxX_axis = maxX_axis
        self.minY_axis = minY_axis
        self.maxY_axis = maxY_axis
        self.saveFlag = saveFlag
        self.figName = figName
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legends = legends
        self.showFlag = showFlag
        self.figsize = figsize
        self.dpi = dpi
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.ColorList = ColorList
        self.palette_cmap = palette_cmap
        self.reorder_num_cluster = reorder_num_cluster
        self.heatmap_reorder = heatmap_reorder
        self.paper_rc = paper_rc
        self.aspect = aspect
    
class SparsityCore(object):
    def __init__(self, vec_all=None, init=None, end=None, KernelBW=None, dim=None):
        if(init is None):
            if(False):
                vec_all.sort()
                plt.figure()
                ax = plt.gca()
                plot_hist(vec_all, ax=ax, plot_core=None)
                plt.show()
            std = np.std(vec_all)
            mu = np.mean(vec_all)
            numStd = 4
            init = vec_all.min() - KernelBW*numStd*dim # np.abs(vec_all.min()*5) # 
            end  = vec_all.max() + KernelBW*numStd*dim # np.abs(vec_all.max()*5) # 
            if(std<3 or (mu-numStd*std<init and mu+numStd*std>end)):
                print('')
#                 print('*** start={}, end={}, sparsity mode: cover the whole range'.format(init, end))
            else:
                init = mu-numStd*std
                end = mu+numStd*std
#                 print('*** start={}, end={}, sparsity mode: +-{}*std={}'.format(init, end, numStd, std))
        self.init = init
        self.end = end
        self.KernelBW = KernelBW
        
    
    
def sparsity_eval(X, sparsity_core):
    return np.array([Gdisparity_function(np.squeeze(X[i,:]), sparsity_core) for i in np.arange(X.shape[0])])
    
def HoyerSparsity(x, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to Hoyer is not one array or one vector :(')
    x = np.abs(x.flatten())
    size = x.size
    sparsity = (np.sqrt(size) - np.sum(x)/np.sqrt(np.sum(x**2)))/(np.sqrt(size) -1) 
    return sparsity

def minusLogGdisparity(x, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to minusLog is not one array or one vector :(')
    x = np.abs(x.flatten())
    sparsity = -np.sum(np.log(1+x**2))
    if(clon_inv):
        sparsity /= x.size
    return sparsity
    

def k4Sparsity(x, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to LpNorm is not one array or one vector :(')
    x = np.abs(x.flatten())
    sparsity = np.sum(x**4)/(np.sum(x**2))**2
    if(clon_inv):
        sparsity *= x.size
    return sparsity

def l2l1Sparsity(x, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to l2l1 is not one array or one vector :(')
    x = np.abs(x.flatten())
    sparsity = np.sqrt(np.sum(x**2))/np.sum(x)
    if(clon_inv):
        sparsity *= np.sqrt(x.size)
    return sparsity

def GiniIndex(x):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to GiniIndex is not one array or one vector :(')
    x = np.abs(x.flatten())
    size = x.size
    c = np.sort(x)
    return 1-2*np.sum(np.multiply(c,(size-(np.arange(size)+1)+1/2)/size)) /np.linalg.norm(x, ord=1)
    
    
def LpSparsity(x, p=1, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to LpNorm-function is not one array or one vector :(')
    x = np.abs(x.flatten())
    if(p == 0):
        sparsity = np.sum(x==0)
        if(clon_inv):
            sparsity = sparsity/x.size
    elif(p>0 and clon_inv):
        sparsity = -(np.sum(x**p)/x.size)**(1/p)
    elif(p>0 and not clon_inv):
        sparsity = -np.sum(x**p)**(1/p)
    return sparsity

def LeSparsity(x, th, clon_inv=True):
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to LeNorm-function is not one array or one vector :(')
    x = x.flatten() 
    sparsity = np.sum(np.abs(x)<=th)
    if(clon_inv):
        sparsity /= x.size
    return sparsity
    
    

def Gdisparity_function(x, sparsity_core): 
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to Gdisparity-function is not one array or one vector :(')
    x = x.flatten() # x.shape
    D = x.size
    a = sparsity_core.init
    b = sparsity_core.end
    m = np.sort(x) # m.shape
    # first m index inside [a,b]
    k = int(np.argwhere(m-a >= 0)[0] if m[0] < a else 0)
    # last m index inside [a,b] 
    kPrime = int(np.argwhere(m-b > 0)[0]-1 if m[-1] > b else D-1)
    shared_coeff = 2*(b-a)*sparsity_core.KernelBW**2
    LHS = ((-(a-m[k])**3/3) if (k==0 or a>(m[k-1]+m[k])/2) else ((m[k]-m[k-1])**3/12-(a-m[k-1])**3/3))/shared_coeff
    RHS = (((m[kPrime+1]-m[kPrime])**3/12-(b-m[kPrime+1])**3/3) if (kPrime<D-1 and b>=(m[kPrime]+m[kPrime+1])/2) else ((b-m[kPrime])**3/3))/shared_coeff
    MHS = ((np.sum(np.diff(m[k:kPrime+1])**3))/12)/shared_coeff
    IHS = - np.log((b-a)/(np.sqrt(2*np.pi)*sparsity_core.KernelBW*D))
    S =  IHS + (MHS + LHS + RHS)
    return S
    
    
    
# class SignalGen(Task):        
#     def run(self, load_Core):
#         if('ECoG' in self.task_core.signal_mode) :
#             num_nodes, dimArray, data_load_core = data_load(self.task_core).run(load_Core) # np.any(data_load_core.y_train<0)
#         X = data_load_core.X_train
#         y = data_load_core.y_train 
#         if(data_load_core.X_test is not None):
#             X = np.concatenate((X, data_load_core.X_test),0)
#             y = np.concatenate((y, data_load_core.y_test),0)
#         X, y, clip_sizes, conv_sizes, dimArray = PreProcessing(X, y, None, data_load_core.clip_sizes_train, data_load_core)
#         
#         
#         idx_nonszr = np.argwhere(y==0)[:,0]
#         if(self.task_core.num_nonszr_samp is not None):
#             idx_nonszr = np.random.choice(idx_nonszr, size=np.min([self.task_core.num_nonszr_samp, idx_nonszr.size]), replace=False)
#         try:
#             idx_szr = np.argwhere(y>0) [:,0] # idx_szr.shape
#             idx_preszr = np.argwhere(y<0)[:,0]
#             if(self.task_core.num_szr_samp is not None):
#                 idx_szr = np.random.choice(idx_szr, size=np.min([self.task_core.num_szr_samp, idx_szr.size]), replace=False)
#             if(self.task_core.num_preszr_samp is not None):
#                 idx_preszr = np.random.choice(idx_preszr, size=np.min([self.task_core.num_preszr_samp, idx_preszr.size]), replace=False)
#         except:
#             print('')
#         idx = np.concatenate((idx_nonszr[:,np.newaxis], idx_szr[:,np.newaxis], idx_preszr[:,np.newaxis]), 0)[:,0]
#         X = X[idx]
#         y = y[idx]
#             
#         
#         return X, y, {'idx_nonszr':idx_nonszr, 'idx_szr':idx_szr, 'idx_preszr':idx_preszr}

def rand_choose_diff_int(start, end, size):
    if(end-start>=size):
        return np.sort(np.random.choice(np.arange(start, end, 1), size=(size,), replace=False))
    else:
        return np.sort(np.random.choice(np.arange(start, end, 1), size=(size,), replace=True))

def int_normalize2Sum(arr, norm):
    return (np.floor(norm*arr/np.sum(arr))).astype(int)
    
def sliding_window(a, winSize, strideSize): # TODO check the padding
    sh = (a.size - winSize + 1, winSize)
    st = a.strides * 2
    windows = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::strideSize]
    if(windows[-1,-1] < a.size-1): # TODO make padding more efficient
        padding = np.ones((1, winSize))*(a.size-1)
        counter = 0
        for idx in (np.arange(a.size-1-windows[-1,-1]) + windows[-1,-1] + 1):
            padding[0,counter] = idx
            counter += 1 
        windows = np.concatenate((windows, padding), axis=0)
    return windows.astype(int)


    
def linCoarseningMatGen(V, N, mode):
#     if(mode.lin_regularity == 1):   # => all I_i the same  = int(V/N)
#         I = int(V/N) * np.ones((N,)) 
#     elif(mode.lin_regularity == 0): # => all I_i different => 
#         if(N(N+1)/2 < V):
#             I = np.array([np.arange(N-1)+1 , V-int((N-1)*N/2)])
#         else:
#             I = rand_choose_diff_int(1, V, N)
#             I = int_normalize2Sum(I, V)
#     else:

    if(mode.overlap == 0): #TODO check with toy examples,  for extreme cases etc.
        n_same = int(mode.lin_regularity*N)
        norm_same = int(mode.lin_regularity*V)
        if(n_same>0):
            I_same = ((norm_same/n_same)*np.ones((n_same,))).astype(int)
        else:
            I_same = []
        I_diff = rand_choose_diff_int(1, int(np.floor((-1+np.sqrt(1+8*(V-norm_same)))/2)), N-n_same)
        I_diff = int_normalize2Sum(I_diff, V-norm_same) 
        zeroArg = np.argwhere(I_diff==0)
        nonZeroOneArg = np.argwhere(I_diff>1)
        if(np.any(I_diff==0)):
            I_diff[zeroArg] +=1
            I_diff[nonZeroOneArg] -=1
        # np.sum(I_diff)
        # np.sum(int_normalize2Sum(V-norm_same, I_diff)) # np.sum(np.floor((V-norm_same)*I_diff/np.sum(I_diff)))
        I = np.sort(np.concatenate((I_same, I_diff)))
        I[-1] += V-np.sum(I) #if V>np.sum(I) else 0
        indices = (np.concatenate(([0], np.cumsum(I)))).astype(int)
        #TODO make it vectorized and more efficient
        T = np.zeros((N, V))
        for i in np.arange(N): 
            T[i, indices[i]:indices[i+1]] = 1
        
    if(mode.lin_regularity == 1):
        # length = numWin(ws-os) + os - r and numWin=N, length=V => V=N*ws-(N-1)*os -r => V =N(ws)*(1-mode.overlap) + ws*mode.overlap -r
#         previou
#         c = int(np.ceil(V/((N-2)*(1-mode.overlap)+1)))
#         r = V-(N-2)*(c-int(np.ceil(c*mode.overlap)))-c
#         while(r>c-int(np.ceil(c*mode.overlap))):
#             c += 1
#             r = V-(N-2)*(c-int(np.ceil(c*mode.overlap)))-c
#         while(r<=0):
#             c -= 1
#             r = V-(N-2)*(c-int(np.ceil(c*mode.overlap)))-c
#         winSize = c
#         overlapSize = int(np.ceil(winSize*mode.overlap))        
#         
#         indices = sliding_window(np.arange(V), winSize, winSize-overlapSize) # indices.shape
#         if(indices.shape[0]!=N):
#             print('Number of windows is {} not N={} when overlap is {}'.format(indices.shape[0], N, mode.overlap))   #raise ValueError
#             if(indices.shape[0]<N):
#                 padding = (np.ones((N-indices.shape[0], winSize))*(V-1)).astype(int)
#                 indices = np.concatenate((indices, padding), axis=0)
#         T = np.zeros((N, V))
#         for i in np.arange(N):
#             T[i, indices[i,:]] = 1       


        c = int(np.ceil(V/((N-1)*(1-mode.overlap)+1)))
        r = V-(N-1)*(c-int(np.ceil(c*mode.overlap)))
        while(r>c-int(np.ceil(c*mode.overlap))):
            c += 1
            r = V-(N-1)*(c-int(np.ceil(c*mode.overlap)))
        while(r<=0):
            c -= 1
            r = V-(N-1)*(c-int(np.ceil(c*mode.overlap)))
            
        winSize = c
        overlapSize = int(np.ceil(winSize*mode.overlap))        
        
        indices = sliding_window(np.arange(V), winSize, winSize-overlapSize) # indices.shape
        if(indices.shape[0]!=N):
            print('Number of windows is {} not N={} when overlap is {}'.format(indices.shape[0], N, mode.overlap))   #raise ValueError
            if(indices.shape[0]<N):
                padding = (np.ones((N-indices.shape[0], winSize))*(V-1)).astype(int)
                indices = np.concatenate((indices, padding), axis=0)
        T = np.zeros((N, V))
        for i in np.arange(N):
            T[i, indices[i,:]] = 1  



    # np.split(np.arange(V), [3, 5, 6, 10])

#     if(mode.homogeneous):
#         if(mode.overlap):
#             T = np.kron( np.eye(N), np.ones((1, int(V/N))) ) # T.shape
#         elif(not mode.overlap): 
#             raise ValueError('homogeneous-overlapped fine-coarse mapping is not implemented yet :(')  #TODO 
#     elif(not mode.homogeneous):
#         if(mode.overlap):
#             raise ValueError('non-homogeneous-overlap fine-coarse mapping is not implemented yet :(') #TODO 
#         elif(not mode.overlap):
#             raise ValueError('non-homogeneous-overlapped fine-coarse mapping is not implemented yet :(') #TODO 
    if(isinstance(mode.scaling, str)):
        if(mode.scaling == 'row_normalize'):
            row_sums = T.sum(axis=1)
            T = T / row_sums[:, np.newaxis]
        elif(mode.scaling == 'mat_normalize'):
            T = T / T.sum()
    else:
        T = T*mode.scaling
    return T 
    # np.set_printoptions(threshold=sys.maxsize)
            

def graph_reorder(x, num_cluster=4):
    if(np.ndim(x)!=2 or x.shape[0]!=x.shape[1]):
        raise ValueError('Input to graph_reorder is not a matrix :(')
    W = x
    num_nodes = W.shape[0]
    if(False):
#         W = np.reshape(x, (num_nodes,num_nodes))
    #     data, rows, columns = make_checkerboard(shape=(20, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=0)
        model = SpectralCoclustering(n_clusters=num_cluster, random_state=0) # SpectralCoclustering, SpectralBiclustering
        model.fit(W)
        newW = W[np.argsort(model.row_labels_)]
        W = newW[:, np.argsort(model.column_labels_)]
    elif(True):
        W = np.reshape(x, (num_nodes,num_nodes))
        graph = csr_matrix(W)
        perm = reverse_cuthill_mckee(graph)
        W = W[np.ix_(perm, perm)] # W[perm, perm]
    elif(False):
        cluster_sizes = np.array([6, 3, 4, 5]).astype(int)
        block_sizes = np.matmul(cluster_sizes[:,np.newaxis], (cluster_sizes[:,np.newaxis]).T)
        block_sizes[np.diag_indices(num_cluster, 2)] = [int(cluster_sizes[i]*(cluster_sizes[i]+1)/2) for i in np.arange(num_cluster)]
        block_sizes_flat =  block_sizes[np.triu_indices(num_cluster)] # block_sizes.flatten('r')
        order_block_sizes = []
        for i in np.arange(num_cluster):
            for j in np.arange(i, num_cluster):
                order_block_sizes.append(j-i)
        order_block_sizes = np.array(order_block_sizes)
        sorted_block_sizes_flat = block_sizes_flat[np.argsort(order_block_sizes)]   
        sorted_order_block_sizes = np.sort(order_block_sizes)
        sorted_block_sizes_flat = sorted_block_sizes_flat[sorted_order_block_sizes>=0]
        sorted_block_sizes_flat = np.flip(sorted_block_sizes_flat)
        absSortedIdx = np.argsort(np.abs(x))
        cumsum = np.concatenate(([0],np.cumsum(sorted_block_sizes_flat)))
        block_vals = [x[absSortedIdx[cumsum[i]:cumsum[i+1]]] for i in np.arange(cumsum.size-1)]
        # de-sort
        block_vals.reverse()
        de_sort_block_vals = [block_vals[i] for i in np.argsort(order_block_sizes)]
        ## filling
        W = -1e10*np.ones((num_nodes,num_nodes))
        cumsum_cluster = np.concatenate(([0],np.cumsum(cluster_sizes)))
        counter = 0
        for i in np.arange(num_cluster):
            for j in np.arange(i, num_cluster):
                print('i={},j={},counter={}'.format(i,j,counter))
                fill_arr = de_sort_block_vals[counter]
                # fill diagonal blocks
                if(i==j):
                    fill_mat = -1e10*np.ones((cluster_sizes[i], cluster_sizes[i]))
                    fill_mat[np.triu_indices(cluster_sizes[i])] = fill_arr
                    fill_mat[np.tril_indices(cluster_sizes[i])] = fill_arr
                # fill non-diagonal blocks
                else:
                    fill_mat = np.reshape(fill_arr, (cluster_sizes[i],cluster_sizes[j]))
                row_idx = np.arange(cumsum_cluster[i], cumsum_cluster[i+1])
                col_idx = np.arange(cumsum_cluster[j], cumsum_cluster[j+1])
#                 W[row_idx,:][:,col_idx] = fill_mat
#                 W[col_idx,:][:,row_idx] = fill_mat
#                 W[row_idx[:, None], col_idx] = fill_mat
#                 W[col_idx[:, None], row_idx] = fill_mat
                W[np.ix_(row_idx, col_idx)] = fill_mat
                W[np.ix_(col_idx, row_idx)] = fill_mat
                counter += 1
        
    
    return W

def fine_mat_gen(fine_core):
    V = fine_core.size
    W = fine_core.distribution.sample(V)
#     W = graph_reorder(x, fine_core.reorder_num_cluster)
    return W


#  # Add N Obs inside boxplot (optional)
#     def add_n_obs(df, group_col, y):
#         medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}
#         xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
#         n_obs = df.groupby(group_col)[y].size().values
#         for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
#             plt.text(x, medians_dict[xticklabel]*1.01, "#obs : " + str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')
#     
#     #     add_n_obs(df, group_col=groupby_col, y=y_col) 
    
def plot_boxplot(df, groupby_col, target_cols=None, plot_core=None):
    # Draw Plot
    numRows = 1 if len(target_cols)<4 else 3
    fig, axes = plt.subplots(numRows, int(np.ceil(len(target_cols)/numRows)), figsize=(15,10), dpi=80)
    
    for i, y_col in enumerate(target_cols):
        ax=axes.flatten()[i]
        sns.boxplot(x=groupby_col, y=y_col, data=df[[groupby_col,y_col]], notch=False, ax=ax, palette=RColorBrewer_palette[i] ) 
#         sns.color_palette("cubehelix")
#         df.boxplot(y_col, by=groupby_col, ax=ax)

        boxVals = df[[groupby_col, y_col]].groupby(groupby_col)[y_col].apply(list)
        tval, pval = stats.ttest_ind(boxVals[0],boxVals[1])
        ax.title.set_text(('tval={0:.3f},pval={1:.3f}').format(tval, pval))
        
#         As the difference between the sample data and the null hypothesis increases, the absolute value of the t-value increases.
        ax.set_xlabel(groupby_col)
        ax.set_ylabel(y_col)      
        
    # Decoration
    plt.title('' if plot_core is None else plot_core.title) #, fontsize
#     plt.ylim(10, 40)
#     fig.delaxes(axes[1,3]) # remove empty subplot
    plt.tight_layout() 
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()

    
def plot_single_regression(df, x_col, y_col, hue_col,  plot_core=None):
    sns.lmplot(x=x_col, y=y_col, hue=hue_col, data=df, fit_reg=True)
    plt.title('' if plot_core is None else plot_core.title) #, fontsize
#     plt.tight_layout() 
    if(plot_core.saveFlag):
        plt.savefig(plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
        
def relPlot(df, x_col, y_col, hue_col=None, style_col=None, plot_core=None):
#     f, ax = plt.subplots()
#     if(plot_core.log_scale):
#         ax.set(yscale="log")
    sns.set(font_scale=1.3) 
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context(rc = plot_core.paper_rc)  # "paper", 
#     fig, ax = plt.subplots()
    grid = sns.relplot(data=df, x=x_col, y=y_col, hue=hue_col, kind="line", style=style_col, legend='full', \
                                        markers=True, palette=plot_core.palette_cmap, height=6, aspect=plot_core.aspect) # , ci="sd" # , _legend_out=False
    # palettes = ['BuPu', 'hot', sns.color_palette("mako_r", 6) , "tab10", 'RdYlBu' , 'Paired' , 'Dark2', "Reds", sns.cubehelix_palette(8)]
    grid._legend_out = False
    leg = grid._legend
    leg.set_bbox_to_anchor([1, 0.66])  # coordinates of lower left of bounding box
    leg._loc = 4  # if required you can set the loc
    if(plot_core.log_scale):
        grid.set(yscale="log")
    if(plot_core.minX_axis is not None and plot_core.maxX_axis is not None):
        plt.xlim(plot_core.minX_axis, plot_core.maxX_axis)
    if(plot_core.minY_axis is not None and plot_core.maxY_axis is not None):
        plt.ylim(plot_core.minY_axis, plot_core.maxY_axis)
    plt.title('' if plot_core is None else plot_core.title) #, fontsize
#     fig.set_size_inches(plot_core.figsize)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.legend(loc='upper right')
    plt.tight_layout() 
    if(plot_core.saveFlag):
        plt.savefig(plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
        
def plot_regression(df, groupby_col, target_cols=None, plot_core=None):
    numRows = 1 if len(target_cols)<4 else 3
    fig, axes = plt.subplots(numRows, int(np.ceil(len(target_cols)/numRows)), figsize=(15,10), dpi=80)
    dfgroupedby = df.groupby(groupby_col)[target_cols].mean()
    x = np.array(dfgroupedby.index)
    for i, y_col in enumerate(target_cols):
        try:
            ax=axes.flatten()[i]
        except:
            ax = axes
        y = np.array(dfgroupedby[y_col])
        sns.regplot(x=x, y=y, ax=ax)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        ax.title.set_text(('rsquared={0:.3f},pval={1:.3f}').format(r_value**2, p_value))
#         1) low R-square and low p-value (p-value <= 0.05)
#         2) low R-square and high p-value (p-value > 0.05)
#         3) high R-square and low p-value
#         4) high R-square and high p-value
#         Interpretation:
#         1) means that your model doesn't explain much of variation of the data but it is significant (better than not having a model)
#         2) means that your model doesn't explain much of variation of the data and it is not significant (worst scenario)
#         3) means your model explains a lot of variation within the data and is significant (best scenario)
#         4) means that your model explains a lot of variation within the data but is not significant (model is worthless)
        ax.set_xlabel(groupby_col)
        ax.set_ylabel(y_col)
        
    plt.title('' if plot_core is None else plot_core.title) #, fontsize
    plt.tight_layout() 
    if(plot_core.saveFlag):
        plt.savefig(plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
        
def plot_line(df, x_col, target_cols, groupby_col=None, hue_col=None, hue_col2=None, plot_core=None):
    fig, ax = plt.subplots()
    if(groupby_col is not None):
        df = df[[x_col] + target_cols + [groupby_col] + [hue_col]]
    elif(hue_col is not None):
        df = df[[x_col] + target_cols + [hue_col]]
    elif(hue_col2 is not None):
        df = df[[x_col] + target_cols + [hue_col]+ [hue_col2]]
    else:
        df = df[[x_col] + target_cols]
#     plt.figure(figsize=plot_core.figsize, dpi=plot_core.dpi)
#     dfgroupedby = df.groupby(x_col)[target_cols].mean()
#     x = np.array(dfgroupedby.index)
#     palettes = ['BuPu', 'hot']
#     marker_arr = [False, True]
#     for counter, groupby_col in enumerate(groupby_cols):
#         for y_col in target_cols:
#             indf = df[[x_col,y_col,groupby_col]]
#             new_df = indf.melt(id_vars=[x_col,groupby_col], value_name=y_col, var_name='mode')
#             if(plot_core.log_scale):
#                 new_df[y_col] = new_df[y_col].apply(np.log)
#                 new_df = new_df.rename(columns={y_col:'log-'+y_col})
#                 y_col = 'log-'+y_col
#             sns.lineplot(x=x_col, y=y_col, hue=groupby_col, data=new_df, markers=marker_arr[counter]) # , palette = palettes[counter]
    if(groupby_col is not None):
        y_col = plot_core.ylabel
        df = df.melt(id_vars=[x_col,groupby_col], value_vars= target_cols, value_name=y_col, var_name='mode')
        sns.lineplot(x=x_col, y=y_col, hue=groupby_col, data=df) #  , style='mode'
    elif(len(target_cols)==1):
        y_col = target_cols[0]
#         if(plot_core.log_scale):
#             df['log '+y_col] = df[y_col].apply(np.log)
#             df = df.drop(columns=[y_col])
#             y_col = 'log '+ y_col
#         df[hue_col] = df[hue_col].astype('category')
        sns.lineplot(x=x_col, y=y_col, hue=hue_col, data=df, legend='full', style=hue_col if hue_col2 is None else hue_col2, \
                     markers=True, linewidth=4, markersize=10, palette=customPalette1)  # "Paired"

       
    

    
#     var_name = 'mode'
#     value_name = 'sparsity'
#     new_df = df.melt(id_vars=x_col, value_name=value_name, var_name=var_name)
#     sns.lineplot(x=x_col, y=value_name, hue=var_name, data=new_df, palette = "hot", dashes = False, markers = ["o", "<"],  legend="brief")
    plt.rcParams.update({'font.size': 30, 'font.weight':'bold', 'font.family':'normal'})
    plt.rcParams.update({'xtick.labelsize' : 30}) 
    plt.rcParams.update({'ytick.labelsize' : 30}) 
    sns.set_style('ticks')
    sns.set(font_scale = 4)
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    fig.set_size_inches(8, 7)
    if(plot_core.log_scale):
        ax.set_yscale('log')
    xmin, xmax, ymin, ymax = plt.axis()
    ax.set_xticks(np.arange(xmin, xmax, ((xmax-xmin)/10)))
#     ax.set_yticks(np.arange(ymin, ymax, ((ymax-ymin)/10)))
    plt.title('' if plot_core is None else plot_core.title) 
    plt.tight_layout()
#     axes = plt.gca()
#     axes.set_xlim([plot_core.minX_axis, plot_core.maxX_axis])
#     axes.set_ylim([plot_core.minY_axis, plot_core.maxY_axis])
    x1, x2, y1, y2 = plt.axis()
    
    if(plot_core.minY_axis is not None):
        y1 = plot_core.minY_axis
    if(plot_core.maxY_axis is not None):
        y2 = plot_core.maxY_axis
    plt.axis((x1,x2,y1,y2))
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
        
def plot_scatter(df, x_col, y_col, hue_col=None, style=None, plot_core=None):
    plt.figure(figsize=plot_core.figsize, dpi=plot_core.dpi)
    ax = plt.gca()
    df = df[[x_col, y_col] if hue_col is None else [x_col, y_col, hue_col]]
#     df[target_cols[0]] = df[target_cols[0]].astype('int')
    if(plot_core.log_scale):
#         ax.set_yscale('log')
        df['log-'+y_col] = df[y_col].apply(np.log)
        df.drop(columns=[y_col])
        y_col = 'log-'+y_col
    if(plot_core.log_xscale):
#         ax.set_xscale('log')
        df['log-'+x_col] = df[x_col].apply(np.log)
        df.drop(columns=[x_col])
        x_col = 'log-'+x_col
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col, style=style, s=140,\
                                    markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'),\
                                        palette='Paired', legend="full", ax=ax) 
#     ax.legend(loc='lower right')
#     sns.set_context(font_scale=3)
#     plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
#     plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
     
    ax.set_xlabel(x_col, fontsize=16);
    ax.set_ylabel(y_col, fontsize=16);
    
    #    sns.hls_palette(np.unique(df[target_cols[0]].values).size, l=.3, s=.8) # 'RdYlBu' # 'Paired' # 'Dark2'
 
    plt.title('' if plot_core is None else plot_core.title) 
#     plt.tight_layout()
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
    
def plot_graphScaling(df, plot_core):
    fig, axs = plt.subplots(df.shape[0]+1, 2, figsize=plot_core.figsize, dpi=plot_core.dpi)
    if(plot_core.match_ranges):
        vec_all = np.concatenate((np.concatenate([np.reshape(W, -1) for W in df['fine-W'].values])\
                                               , np.concatenate([np.reshape(W, -1) for W in df['coarse-W'].values])))
        plot_core.minX_axis = vec_all.min()
        plot_core.maxX_axis = vec_all.max()
    plot_heatmap(df['fine-W'].iloc[0], axs[0,0], plot_core)
    plot_hist(df['fine-W'].iloc[0], axs[0,1], plot_core)
    for i in np.arange(df.shape[0]):
        plot_heatmap(df['coarse-W'].iloc[i], axs[i+1,0], plot_core)
        plot_hist(df['coarse-W'].iloc[i], axs[i+1,1], plot_core)
        
    plt.title('' if plot_core is None else plot_core.title)
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    plt.show()



def plot_hist_text(df, hist_col, text_cols, remove_str='', plot_core=None):
    fig, axes = plt.subplots(df.shape[0], 1, figsize=plot_core.figsize, dpi=plot_core.dpi)
    if(plot_core.match_ranges):
        vec_all = np.concatenate([np.reshape(W, -1) for W in df[hist_col].values])
        plot_core.minX_axis = vec_all.min()
        plot_core.maxX_axis = vec_all.max()
    for i in np.arange(df.shape[0]):
        ax = axes[i]
        plot_core.facecolor = ColorList[i] if plot_core.ColorList is None else plot_core.ColorList[i]
        plot_hist(df[hist_col].iloc[i], ax, plot_core, axLabel=False)
        vals = list(df[text_cols].iloc[i].values)
        label = ''
        for i, val in enumerate(vals):
            label = label + text_cols[i].replace(remove_str,'') + (\
                                            '={0:.1f}\n'.format(val) if np.abs(val)<1e-5 else \
                                            '={0:.2f}\n'.format(val) if np.abs(val)>0.1 else \
                                            '={0:.3f}\n'.format(val) if np.abs(val)>0.01 else\
                                            '={0:.4f}\n'.format(val) if np.abs(val)>0.001 else\
                                            '={0:.5f}\n'.format(val) )
        ax.text(1.05, 0.5, label, rotation=0, size=10, ha='left', va='center', transform=ax.transAxes)
    plt.title('' if plot_core is None else plot_core.title) 
    plt.tight_layout()
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
  

def plot_heat(df, target_cols, plot_core):
    fig, axes = plt.subplots(df.shape[0], len(target_cols), figsize=plot_core.figsize, dpi=plot_core.dpi)
    
    if(df.shape[0]==1 and len(target_cols)==1):
        plot_heatmap(df, axes, plot_core)
        
    if(df.shape[0]==1):
        axes = axes[np.newaxis,:]
        
    if(len(target_cols)==1):
        axes = axes[:,np.newaxis]
    
    for j, col in enumerate(target_cols):
        if(plot_core.match_ranges):
            vec_all = np.concatenate([np.reshape(W, -1) for W in df[col].values])
            match_centered = np.max([np.abs(vec_all.min()), np.abs(vec_all.max())])
            plot_core.minX_axis = -match_centered
            plot_core.maxX_axis = match_centered
#             plot_core.minX_axis = vec_all.min()
#             plot_core.maxX_axis = vec_all.max()
        for i in np.arange(df.shape[0]):
            ax = axes[i,j]
            plot_heatmap(df[col].iloc[i], ax, plot_core)
                        
    plt.title('' if plot_core is None else plot_core.title) 
#     plt.tight_layout()
#     fig.subplots_adjust(top=0.95) # left=0.15, 
    cols = target_cols # ['Column {}'.format(col) for col in range(1, 4)]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    rows = [] # ['Row {}'.format(row) for row in ['A', 'B', 'r', 'D']]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()

  
            
def plot_heat_text(df, target_cols, text_cols, remove_str, plot_core):
    fig, axes = plt.subplots(df.shape[0], len(target_cols), figsize=plot_core.figsize, dpi=plot_core.dpi)
    for j, col in enumerate(target_cols):
        if(plot_core.match_ranges):
            vec_all = np.concatenate([np.reshape(W, -1) for W in df[col].values])
            match_centered = np.max([np.abs(vec_all.min()), np.abs(vec_all.max())])
            plot_core.minX_axis = -match_centered
            plot_core.maxX_axis = match_centered
#             plot_core.minX_axis = vec_all.min()
#             plot_core.maxX_axis = vec_all.max()
        for i in np.arange(df.shape[0]):
            ax = axes[i,j]
            plot_heatmap(df[col].iloc[i], ax, plot_core)
            if(len(text_cols[j])>0):
                text_col = df[text_cols[j]].iloc[i]
                vals = list(text_col.values)
                label = ''
                for k, val in enumerate(vals):
                    label = label + text_col.index[k].replace(remove_str,'') + (\
                                                    '={0:.1f}\n'.format(val) if np.abs(val)<1e-5 else \
                                                    '={0:.2f}\n'.format(val) if np.abs(val)>0.1 else \
                                                    '={0:.3f}\n'.format(val) if np.abs(val)>0.01 else\
                                                    '={0:.4f}\n'.format(val) if np.abs(val)>0.001 else\
                                                    '={0:.5f}\n'.format(val) )
                ax.text(1.05, 0.5, label, rotation=0, size=10, ha='left', va='center', transform=ax.transAxes)
        
            
    plt.title('' if plot_core is None else plot_core.title) 
    plt.tight_layout()
    fig.subplots_adjust(top=0.95) # left=0.15, 
    cols = target_cols # ['Column {}'.format(col) for col in range(1, 4)]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    rows = [] # ['Row {}'.format(row) for row in ['A', 'B', 'r', 'D']]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()

  
def plot_heat_hist_text(df, target_cols, plot_core, heat_flags, hist_flags, text_flags, text_cols=None, add_remove_text_cols=None):
    fig, axes = plt.subplots(df.shape[0], sum(heat_flags)+sum(hist_flags)+sum(text_flags), figsize=plot_core.figsize, dpi=plot_core.dpi)
    Colax_counter = 0
    label_cols = []
    for j, col in enumerate(target_cols):
        if(plot_core.match_ranges):
            vec_all = np.concatenate([np.reshape(W, -1) for W in df[col].values])
            match_centered = np.max([np.abs(vec_all.min()), np.abs(vec_all.max())])
            plot_core.minX_axis = -match_centered
            plot_core.maxX_axis = match_centered
#             plot_core.minX_axis = vec_all.min()
#             plot_core.maxX_axis = vec_all.max()
        if(heat_flags[j]):
            heat_ax = Colax_counter
            Colax_counter += 1
            label_cols.append(col)
            
        if(hist_flags[j]):
            hist_ax = Colax_counter
            Colax_counter += 1
            label_cols.append('-> Histogram')
        
        if(text_flags[j]):
            text_ax = Colax_counter
            Colax_counter += 1
            label_cols.append('-> Sparsity')
            
        for i in np.arange(df.shape[0]):
            if(heat_flags[j]):
                plot_heatmap(df[col].iloc[i], axes[i,heat_ax], plot_core)
                
            if(hist_flags[j]):
                plot_hist(df[col].iloc[i], axes[i,hist_ax], plot_core)
               
            if(text_cols is not None and text_flags[j]):
                add_remove_str = add_remove_text_cols[j]+'-'
                text_col = df[[add_remove_str+text_cols[j][k] for k in np.arange(len(text_cols[j]))]].iloc[i]
                vals = list(text_col.values)
                label = ''
                for k, val in enumerate(vals):
                    label = label + text_col.index[k].replace(add_remove_str,'') + (\
                                                    '={0:.1f}\n'.format(val) if np.abs(val)<1e-5 else \
                                                    '={0:.2f}\n'.format(val) if np.abs(val)>0.1 else \
                                                    '={0:.3f}\n'.format(val) if np.abs(val)>0.01 else\
                                                    '={0:.4f}\n'.format(val) if np.abs(val)>0.001 else\
                                                    '={0:.5f}\n'.format(val) )
                axes[i,text_ax].set_axis_off()
                plt.text(0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', \
                                transform=axes[i,text_ax].transAxes, fontsize=12) # , color='r', bbox=dict(facecolor='red', alpha=0.5)
                          
#             if(text_cols is not None and len(text_cols[j])>0):
#                 ax.text(1.05, 0.5, label, rotation=0, size=10, ha='left', va='center', transform=ax.transAxes)
        
            
    plt.title('' if plot_core is None else plot_core.title) 
#     plt.tight_layout()
    fig.subplots_adjust(top=0.95) # left=0.15, 
    # ['Column {}'.format(col) for col in range(1, 4)]
    for ax, col in zip(axes[0], label_cols):
        ax.set_title(col)
    rows = [] # ['Row {}'.format(row) for row in ['A', 'B', 'r', 'D']]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large')
    
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
        
        
def plot_array(t, arrays, plot_core=None):
    figsize = (10, 8)
    dpi = 80
    facecolor=columnToApply
    edgecolor='k'
#     shapes = ['rs--', 'bs-', 'g^-', 'yo--']
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
#     ax = fig.add_subplot(2, 1, 1)
    for i in np.arange(len(arrays)):
        plt.plot(t, arrays[i]) # , shapes[i]
    if(plot_core.legends is not None):
        plt.legend(plot_core.legends)
    plt.xlabel(plot_core.xlabel)
    plt.ylabel(plot_core.ylabel)
    if(plot_core.log_scale):
        plt.yscale('log')
    plt.title('' if plot_core is None else plot_core.title)
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    plt.show()
    

def plot_graph_density(W_tilde, W=None, plot_core=None, extra_heatMat_arr=None, extra_hist_flag=False ):
#     matplotlib.get_backend()
#     plt.ion()
    if(W is not None):
        fig, axs = plt.subplots(2, 2, figsize=plot_core.figsize, dpi=plot_core.dpi, facecolor='w', edgecolor='k')
        if(plot_core.match_ranges):
            minX_axis = np.min([W_tilde.min(), W.min()])
            maxX_axis = np.max([W_tilde.max(), W.max()])
            plot_core.minX_axis = minX_axis
            plot_core.maxX_axis = maxX_axis
        plot_core.fineGraphFlag = True
        plot_heatmap(W, axs[0,0], plot_core)
        plot_hist(W, axs[0,1], plot_core)
        plot_core.fineGraphFlag = False
        plot_heatmap(W_tilde, axs[1,0], plot_core)
        plot_hist(W_tilde, axs[1,1], plot_core)
        
    else:
        fig, axs = plt.subplots(len(W_tilde), 2 if extra_heatMat_arr is None else 4 if extra_hist_flag else 3,\
                                 figsize=plot_core.figsize, dpi=plot_core.dpi, facecolor=plot_core.facecolor, edgecolor=plot_core.edgecolor)
        if(len(W_tilde)==1):
            axs = axs[np.newaxis,:]
        for i, A_in in enumerate(W_tilde):
            plot_heatmap(A_in, axs[i,0], plot_core)
            plot_hist(A_in, axs[i,1], plot_core)
            
    title_str = ' '
    fig.suptitle(title_str)  # , fontsize=16
    if(plot_core.saveFlag):
        plt.savefig( plot_core.figName + '.png')
    if(plot_core.showFlag):
        plt.show()
  
def plot_heatmap(x, ax, plot_core=None, cmap='RdBu_r'):
    if(plot_core is not None and plot_core.palette_cmap is not None):
        cmap = plot_core.palette_cmap
#     if(x.min()>=0):
#         cmap = "Reds" # sns.cubehelix_palette(8)
        
    if(np.unique(x).size<3):
#         colors = ['skyblue', 'darkred'] # ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
#         cmap = sns.xkcd_palette(colors)
        cmap = "Reds"
        plot_core.minX_axis = x.min()  
        plot_core.maxX_axis = x.max()+0.5
        
    x = np.array(x)
#     if(plot_core.heatmap_reorder):
#         x = graph_reorder(x)
    sns.heatmap(x, vmin=plot_core.minX_axis, vmax=plot_core.maxX_axis, cmap=cmap, ax=ax) #, center=0 , cmap='Set1'
#     np.random.choice(np.arange(len(RColorBrewer_palette)))
    
def plot_hist(x, ax, plot_core=None, axLabel=True):
    x = np.reshape(np.array(x), -1)
    num_bins = 80 #int(x.size/50)
    if(plot_core is not None):
        if(plot_core.num_bins is not None):
            num_bins = plot_core.num_bins
        if(plot_core.log_scale == True):
            raw_bins = np.arange(np.min(x), np.max(x), step=(np.max(x)-np.min(x))/num_bins)
            # histogram on log scale. 
            # Use non-equal bin sizes, such that they look equal on log scale.
            bins = np.logspace(np.log(raw_bins[0]), np.log(raw_bins[-1]), len(raw_bins))
    #         if(minX_axis is None):
    #             minX_axis = np.log10(minX_axis)
    #             maxX_axis = np.log10(maxX_axis)
        else:
            bins = num_bins
        if(plot_core.minX_axis is None or not plot_core.match_ranges): 
            # Density Plot and Histogram of all arrival delays
            sns.distplot(x, hist=True, kde=True, bins=bins, color = 'darkblue', \
                         hist_kws={'edgecolor':plot_core.edgecolor, 'facecolor':plot_core.facecolor}, ax=ax) # , kde_kws={'linewidth': 4}
#             ax.hist(x, bins=bins, normed=1, histtype='stepfilled', facecolor=plot_core.facecolor, edgecolor=plot_core.edgecolor) # , facecolor='blue', alpha=0.5  # x.min() , x.max()
        else:
            sns.distplot(x, hist=True, kde=True, bins=bins, color = 'darkblue',\
                         hist_kws={'edgecolor':plot_core.edgecolor, 'facecolor':plot_core.facecolor}, ax=ax)  # , kde_kws={'linewidth': 4}
#             ax.hist(x, bins=bins, range=(plot_core.minX_axis, plot_core.maxX_axis), normed=1, histtype='stepfilled', \
#                     facecolor=plot_core.facecolor, edgecolor=plot_core.edgecolor) # , alpha=0.5
    #     ax.title.set_text()
    else:
        ax.hist(x, bins=num_bins, normed=1)
#     if(plot_core is not None and axLabel):
#         ax.set_title(label=('Gdisparity={0:.2f}').format(Gdisparity_function(x, sparsity_core=plot_core.sparsity_core)), loc='right')
    #     ax.text(1.05, 0.5, ('Gdisparity={0:.2f}').format(Gdisparity_function(x, sparsity_core=plot_core.sparsity_core)),
    #         rotation=0, size=10, 
    # #         weight='bold',
    # #         bbox=dict(edgecolor='lightgreen', facecolor='none', pad=10, linewidth=3),
    #         ha='left', va='center', transform=ax.transAxes)
    ax.set_xlabel('Edge Weights')
    ax.set_ylabel('histogram')
    

def Schur_complement(Omega, b): 
    # b: samp_set, a: marg_set
    a = np.setdiff1d(np.arange(Omega.shape[0]), b)
    invTerm = np.linalg.pinv(Omega[a[:, None],a]) 
#     np.linalg.cond(Omega[a[:, None],a]) > np.finfo(Omega[a[:, None],a].dtype).eps
#     np.linalg.cond(Omega) > np.finfo(Omega.dtype).eps
    # invTerm[np.where(~np.eye(a.shape[0],dtype=bool))]
    if(np.any(np.isnan(invTerm))):
        raise ValueError('nan value occurred in Schur-complement :(')
    return Omega[b[:, None],b] - np.matmul(Omega[b[:, None],a], np.matmul(invTerm,Omega[a[:, None],b])) 
    # np.all(np.matmul(W[b[:, None],a], np.matmul(np.linalg.inv(W[a[:, None],a]),W[a[:, None],b]))==0)
    # np.all(W[b[:, None],b] == W)

def nonDiagMask(num):
    return np.where(~np.eye(num,dtype=bool))
    
def InvGraphLaplacian(B, diagvals=None):
    W = - B
    num = W.shape[0]
    if(diagvals is None): 
        diagvals = np.random.choice(W[nonDiagMask(num)].flatten(), size=num) # np.zeros((num,)) # np.mean(B, axis=1) #  np.diag(W) # np.sum(B, axis=1) # 
    W[np.diag_indices(num)] = diagvals
    return W

def GraphLaplacian(B):
    return np.diag(np.sum(B,axis=1)) - B


def numCommonElems(ar1, ar2):
#     comElems = np.intersect1d(ar1, ar2)
#     print('common elems between ar1={} and ar2={} is {}, size={}'.format(ar1, ar2, comElems, comElems.size))
    return np.intersect1d(ar1, ar2).size
    


# methods = {'neighborIdx': neighborIdx}

def MapFineToCoarse(fine_graph, measuring_core=None, param=None):
    if(fine_graph.adjacencyMatrix is not None):
        W = fine_graph.adjacencyMatrix
        if(measuring_core is not None):
            if(measuring_core.linCoarseningMat is not None):
                B = measuring_core.linCoarseningMat 
                return np.matmul(np.matmul(B,W), B.T)
            else:
                # TODO can be half calculations
                return np.array([[W[np.array(sense1)[:,None], np.array(sense2)].sum() for sense1 in iter(measuring_core.senseIdx)] for sense2 in iter(measuring_core.senseIdx)])
        if(hasattr(param, 'fine2Coarse_core') and param.fine2Coarse_core.mode.Schur_comp):
            Omega = GraphLaplacian(W)
            subsampleSet = param.fine2Coarse_core.subsample_set
            OmegaTilde = Schur_complement(Omega, subsampleSet) #TODO check here # , diagvals=np.diag(W)
            return InvGraphLaplacian(OmegaTilde, diagvals=None) # =np.diag(W)[subsampleSet]
            # np.all( InvGraphLaplacian(Schur_complement(GraphLaplacian(W), samp_set, marg_set), diagvals=np.diag(W)) == W)
            # np.all( Schur_complement(GraphLaplacian(W), samp_set, marg_set) == GraphLaplacian(W))
            # np.all( InvGraphLaplacian(GraphLaplacian(W)) == W)
        elif(~hasattr(param, 'fine2Coarse_core') or param.fine2Coarse_core.mode.linear):
            B = param.fine2Coarse_core.linear_Coarsening_mat #TODO B or T?? B.shape
        #         B = np.eye(V)
        #         B = B[samp_set,:]
            if(B.shape[0]==W.shape[0]):
                W_tilde = W
            else:
                W_tilde = np.matmul(np.matmul(B,W), B.T) # np.reshape(, ()) # 
    else:
        if(measuring_core.linCoarseningMat is not None):
            B = measuring_core.linCoarseningMat 
            raise ValueError('MapFineToCoarse with matrix B is not implemented :(')
        else:
            # TODO can be half calculations
            
#             if(False):
#                 W_tilde = np.array([[numCommonElems(list(itertools.chain.from_iterable([list(fine_graph.neighborIdx(v)) for v in sense1])),sense2) \
#                                    for sense1 in iter(measuring_core.senseIdx)] \
#                                             for sense2 in iter(measuring_core.senseIdx)])
#             else:
            L = len(measuring_core.senseIdx)
#             print('Calculating neighbourIdx inline...')
#             neighbourIdx = [list(itertools.chain.from_iterable([list(fine_graph.neighborIdx(v)) for v in sense])) for sense in iter(measuring_core.senseIdx)]
#             print('before W-tilde inline calculation...')
#             W_tilde = np.array([[numCommonElems(neighIdx, sense) for neighIdx in neighbourIdx] for sense in iter(measuring_core.senseIdx)])
            
            counter_complete = 0
            def numCommonElems(a, b):
                nonlocal counter_complete
                if(counter_complete%1000 == 0):
                    print('Generating Coarse Network {0:.0%} Completed '.format(counter_complete*2/(L*(L-1))))
                counter_complete += 1
#                FALSEEEEE return len(set(neighbourIdx[a]).intersection(measuring_core.senseIdx[b]))
#                FALSEEEEE return np.intersect1d(neighbourIdx[a], measuring_core.senseIdx[b], assume_unique=False).size
#                FALSEEEEE return len(list((Counter(neighbourIdx[a]) * Counter(measuring_core.senseIdx[b])).elements()))
#                 return len([i for i in neighbourIdx[a] if i in measuring_core.senseIdx[b]])
#                 return sum([Counter(neighbourIdx[a])[i] for i in Counter(measuring_core.senseIdx[b])])
                return sum([len(set(fine_graph.neighborIdx(v)).intersection(measuring_core.senseIdx[b])) for v in measuring_core.senseIdx[a]])
            
            print('Generating itertool pairs...')
            pairs = itertools.combinations(np.arange(L), 2)
            print('Generating itertool res...')
            # @jit(target ="cuda")
            res = dict([ (t, numCommonElems(*t)) for t in pairs])
            print('Generating W-tilde from res...')
            W_tilde = np.zeros((L,L))
            W_tilde[np.triu_indices(L, k=1)] = list(res.values())
            W_tilde += W_tilde.T
            W_tilde[np.diag_indices(L)] = [numCommonElems(a, a) for a in np.arange(L)]
#             print('after W-tilde in line calculation')
            print('Wtilde = ', W_tilde)             
    return W_tilde
        
def cross_spectrum(X, normalize=False):
    poww = np.sqrt(np.sum(np.abs(X)**2, -1))
    poww = np.matmul(np.expand_dims(poww, 3), np.expand_dims(poww, 2))
    arr_real = np.real(X)
    arr_imag = np.imag(X)
    real_part = (np.matmul(arr_real, np.transpose(arr_real, [0,1,3,2])) - np.matmul(arr_imag, np.transpose(arr_imag, [0,1,3,2])))
    imag_part = (np.matmul(arr_real, np.transpose(arr_imag, [0,1,3,2])) + np.matmul(arr_imag, np.transpose(arr_real, [0,1,3,2])))
    W =  real_part + 1j *imag_part
    if(normalize):
        raise ValueError('normalizing cross_spectrum is not implemented yet :(')
        # np.true_divide(np.abs(W), poww) # np.sqrt(real_part**2+imag_part**2)
    else:
        W = np.abs(W) 
    W = np.squeeze(np.mean(W, axis=1)) 
    return W


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, Filter_Properties):
    b, a = butter_bandpass(Filter_Properties.lowcut, Filter_Properties.highcut, Filter_Properties.fs, order=5)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def multisignal_filter(X, Filter_Properties): #TODO must be checked!
    filteredX = [[butter_bandpass_filter(np.squeeze(X[i,j,:]), Filter_Properties) for j in np.arange(X.shape[1])]\
                  for i in np.arange(X.shape[0])]
    return np.array(filteredX)

def signal_filter(X, Filter_Properties): #TODO must be checked!
    return butter_bandpass_filter(X, Filter_Properties)

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def inner_windowing(X):
    out = np.array([[np.array(list(window(np.squeeze(X[i,j,:]), n=3))) for i in np.arange(X.shape[0])] for j in np.arange(X.shape[1])])
    print(np.array(list(window(np.squeeze(X[1,1,:]), n=3))).shape)
    return out

def batched_apply(X, func):
    idx_array = []
    batch_size = 500
    batch_num = 0
    num_samples = X.shape[0]
    while(batch_num * batch_size < num_samples): 
        start_idx = batch_num * batch_size
        batch_num += 1
        end_idx = start_idx + batch_size
        if(end_idx > num_samples):
            end_idx = num_samples
        idx_array.append(np.arange(start_idx, end_idx, 1))
    W = None
    for idx in idx_array:
        W_in =  func(X[idx,...])
        W = W_in if W is None else np.concatenate((W, W_in), axis=0)
    W = np.array(W)
    return W

def band_coherence(x, y, Filter_Properties):
    f, Cxy = signal.coherence(x, y, Filter_Properties.fs)
#     coh_filter_Properties = Filter_Properties
#     coh_filter_Properties.fs = f[-1]
#     np.mean(signal_filter(Cxy, coh_filter_Properties))
    idx = (f>=Filter_Properties.lowcut)*(f<=Filter_Properties.highcut)
    return np.mean(Cxy[np.where(idx)])

def band_cross_spectrum(x, y, Filter_Properties): #TODO double check
    f, Pxy = signal.csd(x, y, Filter_Properties.fs) # 
    idx = (f>=Filter_Properties.lowcut)*(f<=Filter_Properties.highcut)
    return np.abs(np.mean(Pxy[np.where(idx)]))
     
def apply_pairwise(X, func, *args):
    return [[[func(np.squeeze(X[k,i,:]), np.squeeze(X[k,j,:]), *args)\
                for i in np.arange(X.shape[1])] \
                    for j in np.arange(X.shape[1])] \
                        for k in np.arange(X.shape[0])]

def signal2Graph(X, task_core):
    if(X.shape[-1] > 1):
        raise Exception('Data should be raw signal!')
    X = X[:,:,:,0]    
    if('coherence' in task_core.graphL_model):
#         X = multisignal_filter(X, task_core.filters[0])
        W = apply_pairwise(X, band_coherence, task_core.filters[0])
    elif('cross-spectrum' in task_core.graphL_model):
        W = apply_pairwise(X, band_cross_spectrum, task_core.filters[0])
    elif('corr' in task_core.graphL_model):
#         X = np.real(X)
#         W = np.matmul(X, np.transpose(X, [0,2,1]))
#         W = np.array([[[pearsonr(np.squeeze(X[samp,row,:]),np.squeeze(X[samp,col,:]))[1] for row in np.arange(num_nodes)] for col in np.arange(num_nodes)] for samp in np.arange(X.shape[0])])
        W = [pandas.DataFrame(data=np.squeeze(X[samp,:,:])).T.corr().values for samp in np.arange(X.shape[0])]
        del X
    elif('cov' in task_core.graphL_model):
        W = [pandas.DataFrame(data=np.squeeze(X[samp,:,:])).T.cov().values for samp in np.arange(X.shape[0])] 
    elif('invCov' in task_core.graphL_model):
        W = [np.linalg.pinv(pandas.DataFrame(data=np.squeeze(X[samp,:,:])).T.cov().values) for samp in np.arange(X.shape[0])]
        for samp in np.arange(X.shape[0]):
            if not np.isfinite(W[samp]).all():
                raise ValueError("array contains infs or NaNs in invCov")    
    
    W = [graph_reorder(np.array(W_in)) for W_in in W]
    W = np.array(W)
    return W
    
def graphGen_PostProcessing(W):
    triu_indices = np.triu_indices(W.shape[-1])
    return np.array([W[i,triu_indices[0],triu_indices[1]] for i in np.arange(W.shape[0])]) 


def Gsparsity_function(x): 
    x = np.abs(x).flatten()
    if(np.ndim(x)>2 or (np.ndim(x)==2 and x.shape[0]!=x.shape[1])):
        raise ValueError('Input to Gsparsity-function is not one array or one vector :(')
    Gsparsity = stats.skew(x) # stats.logistic.cdf() # (x.max()-x.mean())/(x.max()-x.min())
    return Gsparsity

def calc_all_sparsities(df, KernelBW, columnToApply, ColExtStr='', sparsity_core=None, cloningInvFlag = False):
    if(sparsity_core is None):
        vec_all = np.concatenate([np.reshape(W, -1) for W in df[columnToApply].values])
        dim=np.max([W.shape[0] for W in df[columnToApply].values])
        sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=dim)
    df[ColExtStr+'SparsityL0'] = df[columnToApply].apply(LpSparsity, args=[0, cloningInvFlag])
    df[ColExtStr+'SparsityL1'] = df[columnToApply].apply(LpSparsity, args=[1, cloningInvFlag])
    df[ColExtStr+'SparsityLe'] = df[columnToApply].apply(LeSparsity, args=[0.5, cloningInvFlag])
    df[ColExtStr+'minusLogGdisparity'] = df[columnToApply].apply(minusLogGdisparity, args=[cloningInvFlag])
    df[ColExtStr+'k4Sparsity'] = df[columnToApply].apply(k4Sparsity, args=[cloningInvFlag])
    df[ColExtStr+'l2l1Sparsity'] = df[columnToApply].apply(l2l1Sparsity, args=[cloningInvFlag])
    df[ColExtStr+'Hoyer'] = df[columnToApply].apply(HoyerSparsity, args=[cloningInvFlag])
    df[ColExtStr+'GiniIndex'] = df[columnToApply].apply(GiniIndex)
    df[ColExtStr+'Gdisparity'] = df[columnToApply].apply(Gdisparity_function, args=[sparsity_core])
    df[ColExtStr+'Gsparsity'] = df[columnToApply].apply(Gsparsity_function)
    return sparsity_core, df 

def calc_Gdisparity(df, KernelBW, columnToApply='W', ColExtStr='', sparsity_core=None):
    if(sparsity_core is None):
        vec_all = np.concatenate([np.reshape(W, -1) for W in df[columnToApply].values])
        dim=np.max([W.shape[0] for W in df[columnToApply].values])
        sparsity_core = SparsityCore(vec_all=vec_all, KernelBW=KernelBW, dim=dim)
    df[ColExtStr+'Gdisparity'] = df[columnToApply].apply(Gdisparity_function, args=[sparsity_core])
    return sparsity_core, df 


def graphPruning(W, threshold=None, edgeDensity=None): # TODO check the function
    W = np.abs(W)
    if(np.ndim(W)>2 or (np.ndim(W)==2 and W.shape[0]!=W.shape[1])):
        raise ValueError('Input to graphPruning is an array or vector :(')
    if(edgeDensity is None):
        prunedW = (W>threshold).astype(int)
    else:
        x = np.sort(W.flatten()) # [nonDiagMask(W.shape[0])]
        x = np.flip(x)
        num_keep = int(np.floor(edgeDensity*x.size))
        threshold = x[num_keep]
        prunedW = (W>threshold).astype(int) 
#         indices = (W>threshold)
#         prunedW = np.zeros_like(W)
#         prunedW[indices[0:num_keep,:]] = 1
#     prunedW[np.diag_indices(prunedW.shape[0], 2)] = W[np.diag_indices(W.shape[0], 2)]
    return prunedW
    
    

def minMaxNormalize(X): 
    X = np.abs(X)
    if(np.ndim(X)>2 or (np.ndim(X)==2 and X.shape[0]!=X.shape[1])):
        raise ValueError('Input to graphPruning is an array or vector :(')
    return (X-X.min())/(X.max()-X.min())
    
    
def evalSparsityPreserving(W, Q):
    if(W.shape==Q.shape):
        return np.linalg.norm(W-Q)


def Find_cluster_idx(arr, inArr):
    clust_find = -10*np.ones_like(arr)
    inArr = np.concatenate(([0], inArr))
    for i in np.arange(inArr.size-1): 
        clust_find[(arr<inArr[i+1])*(arr>=inArr[i])] = i
    return clust_find

def geographical2DMesh(V):
    numRows = int(np.floor(np.sqrt(V)))
    numCols = int(np.ceil(V/numRows))
    mesh = np.reshape(np.arange(numRows*numCols), (numRows, numCols))
    return mesh


def count_non_sync(chosen, coverage, sizes_SBM):
    cumsum = np.cumsum(sizes_SBM)
    clust_find_start = Find_cluster_idx(chosen, cumsum)
    clust_find_end = Find_cluster_idx(chosen+coverage-1, cumsum)
    return np.sum(clust_find_start!=clust_find_end)


def choose1DRandomDistant(arr, size, coverage):
    arr = np.array(arr)
    chosen = np.sort(np.random.choice(arr, size, replace=False))
    i = 0
    while(np.all(np.abs(np.diff(chosen))<coverage)):
        if(i%20 == 0):
            print('try # {} to choose {} random numbers with 1D distance {} from array of size={}'.format(i, size, coverage, arr.size))
        chosen = np.sort(np.random.choice(arr, size, replace=False))
        i += 1
    return chosen

def indicesCenteredAround(mesh, rowidx, colidxnp, radius):
    ini_list =  [[mesh[max(min(mesh.shape[0]-1,rowidx+rowShift),0), max(min(mesh.shape[1]-1,colidxnp+colShift),0)] \
                            for rowShift in np.arange(-radius, radius+1)] for colShift in np.arange(-radius, radius+1)]
    return np.unique(np.array(list(chain.from_iterable(ini_list))))

def radiusCoverage(coverageSize):
    radius = 0
    while((radius*2+3)**2<=coverageSize):
        radius += 1
    return radius

def chooseMeshIndexCoverage(mesh, idx, coverageSize):
    rowidx =  np.argwhere(mesh==idx)[0,0]
    colidxnp =  np.argwhere(mesh==idx)[0,1]
    radius = radiusCoverage(coverageSize)
    circleSize = radius*2+1
#     choose all in radius 
    indices = indicesCenteredAround(mesh, rowidx, colidxnp, radius)
#    choose extra indices in the next level (surroundingIndices)
    if(coverageSize-(circleSize)**2>0):
        surroundingIndices = indicesCenteredAround(mesh, rowidx, colidxnp, radius+1)
        remainingIndices = np.random.choice(list(set(surroundingIndices)-set(indices)), size=coverageSize-(circleSize)**2, replace=False)
        indices = np.concatenate((np.array(indices), remainingIndices))
#     counter=0
#     for rowShift in [idx-radius-1, ]:
#         if(counter>=coverageSize-(circleSize-2)^2):
#             return indices
#         indices.append()
    return np.sort(indices)
    
def chooseMeshArrayCoverage(mesh, centers, coverageSize):
    return [chooseMeshIndexCoverage(mesh, idx, coverageSize) for idx in list(centers)]
    
    
def overlapExists(mylist):
    for i in np.arange(len(mylist)):
        for j in np.arange(len(mylist)):
            if(j==i):
                continue
            if(len(list(set(mylist[i]) & set(mylist[j])))>0):
                return True
    return False            
    
    
def choose2DRandomDistant(mesh, outSize, coverageSize):
    mesh = np.array(mesh)
#     sequential choosing
    if(True):
        radius = radiusCoverage(coverageSize)
        intialBannedIdx = []
        for i in list(np.arange(radius))+list(mesh.shape[0]-1-np.arange(radius)):
            for j in np.arange(mesh.shape[1]):
                intialBannedIdx.append(mesh[i,j]) # chooseMeshIndexCoverage(mesh, mesh[i,j], coverageSize)
        for j in list(np.arange(radius))+list(mesh.shape[1]-1-np.arange(radius)):
            for i in np.arange(mesh.shape[0]):
                intialBannedIdx.append(mesh[i,j])
                
        TryCounter = 0
        while(True):
            remaining_indices = list(set(np.arange(mesh.size))- set(intialBannedIdx))
            chosen = []
            if(TryCounter%20 == 0):
                print('try # {} to choose {} random numbers with 2D coverage {} from mesh of shape={}*{}'.format(TryCounter, outSize, coverageSize, mesh.shape[0], mesh.shape[1]))
            for i in np.arange(outSize):
                centerIdx = np.random.choice(remaining_indices, size=1, replace=False)
                
                rowIdx =  np.argwhere(mesh==centerIdx)[0,0]
                colIdx =  np.argwhere(mesh==centerIdx)[0,1]
                
                try:
                    in_Chosen = chooseMeshIndexCoverage(mesh, centerIdx, coverageSize)
                except:
                    break
#                 TODO
#                 choose completely independent rows, or a robust way to minimize the overlap
#                      or len(list(set(remaining_indices)&set(in_Chosen)))<len(in_Chosen)
                if(len(in_Chosen)<coverageSize): # and not ((rowIdx==0 or rowIdx==mesh.shape[0]-1) and (colIdx==0 or colIdx==mesh.shape[1]-1))
                    break
                
                remaining_indices = list(set(remaining_indices)-set(in_Chosen))
                chosen.append(in_Chosen)
            if(len(chosen)==outSize):
                return chosen
            TryCounter += 1

# #     parallel choosing
    else:
        centers = np.sort(np.random.choice(mesh.size, outSize, replace=False))
        chosen = chooseMeshArrayCoverage(mesh, centers, coverageSize)
        TryCounter = 0
        while(overlapExists(chosen)):
            if(TryCounter%20 == 0):
                print('try # {} to choose {} random numbers with 2D coverage {} from mesh of shape={}*{}'.format(TryCounter, size, coverage, mesh.shape[0], mesh.shape[1]))
            centers = np.sort(np.random.choice(mesh.size, size, replace=False))
            chosen = chooseMeshArrayCoverage(mesh, centers, coverageSize)
            TryCounter += 1
    return chosen

def OverlappedCommunities(B, Q_SBM_indices):
    overlapList = []
    for i in np.arange(B.shape[0]):
        inlist = []
        for j in np.arange(len(Q_SBM_indices)):
            if(len(list(set(np.argwhere(B[i,:]!=0).flatten()) & set(Q_SBM_indices[j])))>0):
                inlist.append(j)
        overlapList.append(inlist)        
    return overlapList    
    


def FindMajority(l):
    most_commons = Counter(l).most_common()
    value, count = most_commons[0]
    next_value, next_count = most_commons[1]
    if(count==next_count):
        return None
    return value


class SynchronizationCore():
    def __init__(self, LinCoarsMat_core, SBM_core):
        Q_SBM_indices = SBM_core.comIndices
        total_num_communities = len(Q_SBM_indices)
        overlappedCom = OverlappedCommunities(LinCoarsMat_core.B, Q_SBM_indices)
        self.repCommunity = [FindMajority(coms) for coms in overlappedCom]
        self.syncVec = 1-np.array([len(coms) for coms in overlappedCom])/total_num_communities
        self.syncRatio = np.mean(syncVec)

    
        
        


def sensingLayout(A):
    sumRows = np.sum(A, 0)[:,np.newaxis]
    layout = (np.matmul(sumRows, sumRows.T)!=0).astype(int)
#     layout = np.kron(A, A)
#     layout = np.matmul(A, np.matmul(np.ones((A.shape[1], A.shape[1])), A.T))
    return layout

def coverageRatio(A):
    sumRows = np.sum(A, 0)
    return np.sum(sumRows!=0)/sumRows.size

def SBMCommunityProbGenerator(size, density):
#                 [[0.15, 0.05, 0.92, 0.05, 0.02],
#                  [0.05, 0.15, 0.07, 0.97, 0.07],
#                  [0.92, 0.07, 0.20, 0.07, 0.92],
#                  [0.05, 0.97, 0.07, 0.20, 0.07],
#                  [0.02, 0.07, 0.92, 0.07, 0.30]]
    sparse_ground = nx.to_numpy_matrix(nx.gnp_random_graph(size, density)) # np.zeros((size,size))# 
    sparse_ground[np.diag_indices(size)] = 1
    
    means = np.zeros_like(sparse_ground)
    means[sparse_ground==0] = 0.2
    means[sparse_ground==1] = 0.8

#     Q = np.zeros((size, size))
#     Q[np.diag_indices(size)] = np.random.uniform(0.70, 0.99, size=size)
#     Q[nonDiagMask(size)] = np.random.uniform(0.01, 0.30, size=size*(size-1))

#     Q = means

    Q = np.array([[np.random.normal(loc=means[i,j], scale=0.1, size=1)[0] for i in np.arange(size)] for j in np.arange(size)])
    Q[Q>1] = 1
    Q[Q<0] = 0
    return np.array((Q+Q.T)/2)



class SBMCore():
    def __init__(self, n, K, mesh, density=0.1, LinCoarsMat_core=None, setupSync_core=None):
        self.numNodes = n
        self.numComs = K
        self.Qprobs = SBMCommunityProbGenerator(C, density)
#         np.any(self.Qprobs!=self.Qprobs.T)
        self.P = np.zeros((self.numComs, self.numNodes))
        if(True):
            if(setupSync_core is None):
                self.comSizes = int(np.floor(self.numNodes/self.numComs)) * np.ones((self.numComs,)).astype(int) # [50, 50, 100, 100, 150]
                self.comSizes[-1] = n-np.sum(self.comSizes[:-1])
                self.comIndices = []
                remaining_indices = np.arange(mesh.size)
                for i in np.arange(self.numComs):
                    in_Chosen = np.random.choice(remaining_indices, size=self.comSizes[i], replace=False)
                    self.P[i,in_Chosen] = 1
                    remaining_indices = list(set(remaining_indices)-set(in_Chosen))
                    in_Chosen.sort()
                    self.comIndices.append(in_Chosen)
                    
            else:
                # Step 1: filling the determined fine nodes from the sensing matrix
                for i in np.arange(LinCoarsMat_core.B.shape[0]):
                    B_row = LinCoarsMat_core.B[i,:]
                    I_proto_row = np.argwhere(B_row!=0)
                    splitSizes = np.cumsum(np.round(setupSync_core.syncMat[i,:]*I_proto_row.size).astype(int))
                    splitIdx = np.split(I_proto_row, splitSizes)[:-1]
                    for j, idxSet in enumerate(splitIdx):
                        self.P[j, idxSet] = 1
                # Step 2: get rid of double assigned
                multAssignedNodes = np.argwhere(np.sum(self.P, axis=0)>1)[:,0]
                for i in multAssignedNodes:
                    repComIdx = np.random.choice(np.argwhere(self.P[:,i]!=0)[:,0], size=1, replace=False)
                    self.P[:,i] = 0
                    self.P[repComIdx, i] = 1
                # Step 3: fill the rest 
                ComNonAssNodes = np.argwhere(np.sum(self.P, axis=0)==0)[:,0]
                if(ComNonAssNodes.size>0):
                    comSizesTarget = int(np.floor(self.numNodes/self.numComs))* np.ones((self.numComs,)).astype(int) 
                    comSizesTarget[-1] = self.numNodes-np.sum(comSizesTarget[:-1])
                    preComSizes = np.sum(self.P, axis=1)
#                     TODO handle when comSizesTarget<preComSizes
                    comSizeNeed = (comSizesTarget-preComSizes).astype(int)
                    for i in np.arange(self.numComs):
                        fillIdx = np.random.choice(ComNonAssNodes, size=(comSizeNeed[i],), replace=False)
                        self.P[i, fillIdx] = 1 
                        ComNonAssNodes = list(set(ComNonAssNodes)-set(fillIdx))
#                     take care of the remaining nodes as a result of assigning some fine nodes to zero or multiple communities    
                    ComNonAssNodes = np.argwhere(np.sum(self.P, axis=0)==0)[:,0]
                    if(False): # TODO check ComNonAssNodes.size>0
#                         raise ValueError('Not all fine nodes assigned to communities!')   
                        randComIdxs = np.random.choice(np.arange(self.numComs), size=(ComNonAssNodes.size,))
#                         for i, node in enumerate(ComNonAssNodes):
#                             self.P[comIdx[i], node] = 1
                        multAssignedNodes = np.argwhere(np.sum(self.P, axis=0)>1)[:,0]
                        multAssignedComs = np.argwhere(np.sum(self.P[:,multAssignedNodes], axis=1)!=0)[:,0]
                        for i, node in enumerate(ComNonAssNodes):
                            if(multAssignedComs.size>0):
                                comIdx = multAssignedComs[0]
                                multAssignedNodes = np.argwhere(np.sum(self.P, axis=0)>1)[:,0]
                                multAssignedComs = np.argwhere(np.sum(self.P[:,multAssignedNodes], axis=1)!=0)[:,0]
                            else:
                                comIdx = randComIdxs[i]
                            try:
                                nodeToRemove = list(set(np.argwhere(self.P[comIdx,:]!=0)[:,0]) & set(multAssignedNodes))[0]
                            except:
                                nodeToRemove = list(set(np.argwhere(self.P[comIdx,:]!=0)[:,0]) & set(multAssignedNodes))[0]
                            self.P[comIdx, nodeToRemove] = 0
                            self.P[comIdx, node] = 1
                        
                self.comIndices = [list(np.argwhere(np.squeeze(self.P[j, :])!=0).reshape(-1)) for j in np.arange(self.numComs)]
                self.comSizes = np.sum(self.P, axis=1).astype(int)
                print('Community Sizes in SBM Core ', self.comSizes)
                if(np.any(np.array(aggregateList(self.comIndices))>self.numNodes) or np.sum(self.comSizes)!=self.numNodes):
                    raise ValueError('Invalid community to fine node mapping!')
                
        else:
            sizes_SBM_indices = np.concatenate(([0], np.cumsum(self.comSizes)))
            self.comIndices = [np.arange(sizes_SBM_indices[i],sizes_SBM_indices[i+1]) for i in np.arange(self.numComs)]
        return    
        

class SBMDistProp(object):  
    def __init__(self, SBM_core):
        self.SBM_core = SBM_core
#         self.sizes = SBM_core.comSizes
#         self.probs = SBM_core.Qprobs
#         self.num_communities = SBM_core.C 
#         community_size = int(SBM_core.V/SBM_core.C)
#         self.intra_community = CoarseningParams(fine_core=FineMatCore(size=community_size, distribution= NormalDistProp(5, 1)))
#         self.inter_community = CoarseningParams(fine_core=FineMatCore(size=community_size, distribution= DeltaDistProp(1))) 
        
    def sample(self, size=1, df=None):   # TODO complete
        # from scratch
        if(False): 
            W = np.ones(size) * -100000
            community_size = self.numNodes
            num_communities = int(V/community_size) # param.fine_core.intra_community.V
            for i in np.arange(num_communities):
                idx_1 = np.arange(start=i*community_size, stop=(i+1)*community_size)
                for j in np.arange(num_communities):
                    idx_2 = np.arange(start=j*community_size, stop=(j+1)*community_size)
                    if(i==j):
                        W[idx_1[:,None], idx_2] = fine_mat_gen(self.inter_community)
                    else:
                        W[idx_1[:,None], idx_2] = fine_mat_gen(self.intra_community)   

        # using networkx, very slow 
        if(self.SBM_core.genModule=='networkX'): 
            start_time = datetime.now()
            g = nx.stochastic_block_model(self.SBM_core.comSizes, self.SBM_core.Qprobs, seed=0, selfloops=True) 
    #         np.any(self.SBM_core.Qprobs!=self.SBM_core.Qprobs.T) # , nodelist=aggregateList(self.SBM_core.comIndices)
            print('time elapsed generating networkX SBM {} sec'.format((datetime.now()-start_time).seconds))
            if(False):
                nx.draw(g)
                plt.show()
            comIdx = g.graph["partition"] 
            print('partition sizes in nx.stochastic_block_model generator = ', [len(idd) for idd in comIdx])
            if(False):
                start_time = datetime.now()
                W = nx.to_numpy_matrix(g)
                print('time elapsed converting networkX to W {} sec'.format((datetime.now()-start_time).seconds))
            gg = myGraph(graphP=g, neighborIdx=g.neighbors, communityPartitions=comIdx, adjacencyMatrix=None)
        # using graph_tool, supposed to be faster
        else:
            K = self.SBM_core.Qprobs.shape[0]
            n = self.SBM_core.comSizes.sum()
#             sizes = -1*np.ones((K+1,))
#             sizes[0] = 0
#             sizes[1:] = np.cumsum(self.SBM_core.comSizes)
#             comIdx = [np.arange(sizes[i],sizes[i+1]) for i in np.arange(K)]
            comPartition = np.random.choice(np.arange(K), size=n, replace=True, p=self.SBM_core.comSizes/self.SBM_core.comSizes.sum())
#             print('comIdx generated in graph-tool', comIdx)
            com_connection_sizes = np.array([[int(self.SBM_core.Qprobs[k1,k2]*self.SBM_core.comSizes[k1]*self.SBM_core.comSizes[k2]) for k1 in np.arange(K)] for k2 in np.arange(K)])
#             print('com_connection_sizes = ', com_connection_sizes)
#             comGraph = gt.Graph()
#             comGraph.add_vertex(K)
#             edgePropMap = EdgePropertyMap(pmap, g) 
#             print('gt.adjacency(comGraph , com_connection_sizes) = ', gt.adjacency(comGraph, edgePropMap).T)
            com_connection_sizes += np.diag(np.diag(com_connection_sizes))
            start_time = datetime.now()
            g = gt.generate_sbm(comPartition, com_connection_sizes, out_degs=None, in_degs=None, directed=False) # gt.adjacency(comGraph , com_connection_sizes).T
#           g, comIdx = graph_tool.random_graph(n, lambda: poisson(K), directed=False,
#                         model="blockmodel",
#                         block_membership=lambda: randint(K),
#                         edge_probs=prob)
#             W = spectral.adjacency(g).tolil().rows # graph_tool.spectral.adjacency(g) # g.get_edges([g.edge_index]) # graph_tool.spectral.adjacency(g).tolil()
            print('time elapsed generating graph-tool SBM {} sec'.format((datetime.now()-start_time).seconds))
            start_time = datetime.now()
            if(False):
                W = np.zeros((n,n))
                for v in np.arange(n):
                    W[v, g.get_all_neighbors(v)] = 1
                if(False):
                    print('W generated in graph-tool', W)
                    print('W row sum = ', np.sum(W, 1))
                print('time elapsed converting  graph-tool to W {} sec'.format((datetime.now()-start_time).seconds))
            comIdx = [np.argwhere(comPartition==k) for k in np.arange(K)]
            print('partition sizes in gt.generate_sbm generator = ', [len(idd) for idd in comIdx])
            gg = myGraph(graphP=g, neighborIdx=g.get_out_neighbors, communityPartitions=comIdx, adjacencyMatrix=None)
#         W = reorderMat(W, toPermutation=aggList)
        return gg, df

def aggregateList(myList):
    agg = []
    for inList in myList:
        agg.extend(inList) # .reshape(-1)
    return agg
        
def reorderMat(A, fromPermutation=None, toPermutation=None, dimOnly=None):
    if(toPermutation is not None):
        permIdx = np.empty_like(toPermutation)
        permIdx[toPermutation] = np.arange(len(toPermutation))
#         print('matrix reordering to')
    if(fromPermutation is not None):
#         if(~fromPermutation.isarray):
#             fromPermutation =  fromPermutation.communities
        permIdx = np.array(fromPermutation)
#         print('matrix reordering from')
    if(dimOnly==0 or dimOnly==(dimOnly is None and A.shape[0]==permIdx.size)):
        A = A[permIdx,:]
    if(dimOnly==1 or dimOnly==(dimOnly is None and A.shape[1]==permIdx.size)):
        A = A[:,permIdx]
    
    return A

def applyFuncPandas(df, func, sourceCol, paramCol):
    return [func(df[sourceCol].iloc[i], df[paramCol].iloc[i]) for i in np.arange(df.shape[0])]


class communityRecovery():
    def  __init__(self, B, P):
        self.P = P
        self.B = B
        self.Phi = np.matmul(B, P.T)
        self.Psi = np.kron(self.Phi, self.Phi)
        self.numCom = self.P.shape[0]
        
    def fullObjectiveFunc(self, q):
        q_bar = np.multiply(q, 1-q)      
#         Sigma = np.diag(np.matmul(np.kron(self.P, self.P).T, q_bar))
#         invSigma = np.diag(np.matmul(np.kron(self.P, self.P).T, np.divide(1,q_bar))) 
#         logdetSigma = np.sum(np.log(Sigma)
        Sigma = np.matmul(np.kron(self.B, self.B), np.matmul(np.diag(np.matmul(np.kron(self.P, self.P).T, q_bar)), np.kron(self.B, self.B).T))
        invSigma = np.linalg.pinv(Sigma)
        logdetSigma = np.log(np.linalg.det(Sigma))
        centerOmegaTilde = self.omegaTilde-np.matmul(self.Psi, q)
        return np.matmul((centerOmegaTilde).T, np.matmul(invSigma, centerOmegaTilde)) - logdetSigma
                                   
    def fullOptimize(self, Wtilde):
        self.omegaTilde = np.reshpae(Wtilde, -1)
        q0 = np.random.uniform(size=(self.numCom**2,))
        print('q0 = ', q0)
        # define bounds
        b    = [0.0, 1.0] 
        bounds = [b for _ in np.arange(q0.size)]# (b, b, b, b)
#         bounds = optimize.Bounds([0, -0.5], [1.0, 2.0])
        self.opt_res = optimize.minimize(self.fullObjectiveFunc, x0=q0, bounds=bounds)
#         (objective_func, q0, args, method, jac, hess, hessp, bounds=bounds, constraints, tol, callback, options)
        self.solution = self.opt_res.x


    def relaxedObjectiveFunc(self, q):
        q_bar = np.multiply(q, 1-q)      
        Sigma = np.matmul(np.kron(self.B, self.B), np.matmul(np.diag(np.matmul(np.kron(self.P, self.P).T, q_bar)), np.kron(self.B, self.B).T))
        invSigma = np.linalg.pinv(Sigma)
        logdetSigma = np.log(np.linalg.det(Sigma))
        centerOmegaTilde = self.omegaTilde-np.matmul(self.Psi, q)
        return np.matmul((centerOmegaTilde).T, np.matmul(invSigma, centerOmegaTilde)) - logdetSigma
    
    
    def relaxedOptimize(self, Wtilde, Lambda):
        self.omegaTilde = Wtilde.reshape(-1).T
        innPsi = np.matmul(self.Psi.T, self.Psi)
        self.solution = np.matmul(np.linalg.inv(innPsi+Lambda*np.eye(innPsi.shape[0])), np.matmul(self.Psi.T, self.omegaTilde))
        
        
    def getSolution(self):
        return np.reshape(self.solution, (self.numCom, self.numCom))
    
        
class SyncCore():
    def __init__(self, numCoarseNodes, numCom, syncVals, syncProb, maxCommunityCoverage):
#         syncVals=[1/4, 1/2, 3/4, 1], syncProb=[0.1, 0.3, 0.4, 0.2], maxCommunityCoverage=3
        self.syncVec = np.random.choice(syncVals, size=(numCoarseNodes,), p=syncProb)
        self.coverageSizeVec = np.random.choice(np.arange(2,maxCommunityCoverage+1), size=(numCoarseNodes,))
        self.coverageSizeVec[self.syncVec>=1] = 1
        self.syncMat = np.zeros((numCoarseNodes, numCom))
        for i in np.arange(numCoarseNodes):
            j = i%numCom
            self.syncMat[i,j] = self.syncVec[i]
            otherIdx = np.random.choice(np.setdiff1d(np.arange(numCom), [j]), size=(self.coverageSizeVec[i]-1,))
            if(self.coverageSizeVec[i]>1 and self.syncVec[i]<1):
                self.syncMat[i, otherIdx] = (1-self.syncVec[i])/(self.coverageSizeVec[i]-1)
        return  
    
    def update(self, LinCoarsMat_core, SBM_core, sync_calc_mode='max'):
        self.syncMat = np.matmul(LinCoarsMat_core.B, SBM_core.P.T)
        if(sync_calc_mode == 'max'):
            self.syncVec = np.max(self.syncMat, axis=1)
        elif(sync_calc_mode == 'median'):
            self.syncVec = np.max(self.syncMat, axis=1)
        self.syncRatio = np.mean(np.mean(self.syncVec))
        return
        
def uniqueOrderPreserve(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]   
        
def ElemkMeansRecovery(W, C):
    kmeans = KMeans(n_clusters=int(C*(C+1)/2), random_state=0).fit(np.reshape(W,-1).T) # np.reshape(W,-1) , n_init=30, max_iter=1000
    UQ = kmeans.cluster_centers_
#     UQ = reorderMat(UQ, fromPermutation=uniqueOrderPreserve(np.reshape(kmeans.labels_,(C,C))), toPermutation=None)
    Q = np.zeros((C,C))
    Q[np.triu_indices(C)] = UQ[:,0]
    Q = Q + Q.T
    Q[np.diag_indices(C)] = Q[np.diag_indices(C)]/2
#     Q[np.tril_indices(C)] = UQ[:,0]
    return Q


def Row_Col_kMeansRecovery(W, C):
    
#     row-wise clustering
    kmeans_rows = KMeans(n_clusters=C, random_state=0).fit(W.T)
#     kmeans_rows.cluster_centers_ = reorderMat(kmeans_rows.cluster_centers_, \
#                                               fromPermutation=uniqueOrderPreserve(kmeans_rows.labels_), toPermutation=None)
#     kmeans_rowWise = KMeans(n_clusters=C, random_state=0).fit(kmeans_rows.cluster_centers_.T)
#     averaging
    if(True):
        Q = np.zeros((C,C))
        for i in np.arange(C):
            Q[:,i] = np.mean(kmeans_rows.cluster_centers_[:,kmeans_rows.labels_==i], 1)
            
    Q = reorderMat(Q, fromPermutation=uniqueOrderPreserve(kmeans_rows.labels_))
    return Q
    
    
def PhiPermutationMap(arr):
    idx = np.argmax(arr, 1)  
    return uniqueOrderPreserve(idx)

def communityConnectionProbabilities(g):
    K = len(g.communityPartitions)
    print('Calculating neighbourIdx inline...')
    neighbourIdx = [list(itertools.chain.from_iterable([list(g.neighborIdx(v)) for v in iter(comId.flatten())])) for comId in g.communityPartitions]
    counter_complete = 0
    def commonElems(a, b):
#       return np.intersect1d(neighbourIdx[a], measuring_core.senseIdx[b]).size
        nonlocal counter_complete
        if(counter_complete%100 == 0):
            print('Generating itertool res {0:.0%} Completed '.format(counter_complete/(K*(K-1)/2)))
        counter_complete += 1
        return len(set(neighbourIdx[a]).intersection(g.communityPartitions[b].flatten()))/(K*(K-1)/2)
    
    pairs = itertools.combinations(np.arange(K), 2)
    print('Generating itertool res...')
    res = dict([ (t, commonElems(*t)) for t in pairs])
    q = np.min(list(res.values()))
    p = np.min([commonElems(k,k) for k in np.arange(K)])
    return p, q


labelListFileNames = { 'com-youtube.ungraph': 'com-youtube.all.cmty', # 'com-youtube.top5000.cmty' 
                       'email-Eu-core': 'email-Eu-core-department-labels'}
class SNAPnetworkGen():   
    def __init__(self, mode):  
        self.mode=mode       
        
    def sample(self, size=None, df=None):
#         fh=open('Network dataset/{}.txt'.format(self.mode), 'rb')
#         g=nx.read_edgelist(fh)
#         fh.close()
        g = nx.read_edgelist('Network dataset/{}.txt'.format(self.mode), create_using=nx.Graph(), nodetype=int)
        if(self.mode=='email-Eu-core'):
            with open('Network dataset/{}.txt'.format(labelListFileNames[self.mode])) as f:
                table = pd.read_table(f, sep=' ', header=None, names=['nodeIdx', 'comIdx'])
            comPartition = np.array(table['comIdx'])
            df['n'] = comPartition.size
            df['K'] = np.max(comPartition)
            comIdx = [np.argwhere(comPartition==k) for k in np.arange(df['K'])]
            
        elif(self.mode=='com-youtube.ungraph'):
#             table = np.loadtxt('Network dataset/{}.txt'.format(labelListFileNames[self.mode]), dtype=int)
#             with open('Network dataset/{}.txt'.format(labelListFileNames[self.mode])) as f:
#                 table = pd.read_table(f, sep='\t', header=None)
#             with open('Network dataset/{}.txt'.format(labelListFileNames[self.mode])) as f:
#                 content = f.readlines()
#             # you may also want to remove whitespace characters like `\n` at the end of each line
#             table = [x.strip() for x in content] 
            # open file
            with open('Network dataset/{}.txt'.format(labelListFileNames[self.mode])) as fp:
                # 1. iterate over file line-by-line
                # 2. strip line of newline symbols
                # 3. split line by spaces into list (of number strings)
                # 4. convert number substrings to int values
                # 5. convert map object to list
                comIdx = [list(map(int, line.strip().split('\t'))) for line in fp]
            list(itertools.chain.from_iterable(comIdx))
            df['n'] = np.sum([len(comid) for comid in comIdx])
            df['K'] = len(comIdx)
                  
        print(nx.info(g))
        comSizes = [len(comid) for comid in comIdx]
        print('Community Sizes', )
        df['minCommunitySize'] = np.min(comSizes)
        if(False):
            sp = nx.spring_layout(g)
            plt.axis('off')
            nx.draw_networkx(g, pos=sp, with_labels=False) # node_size=
            plt.show()
#         'email-Eu-core-department-labels.txt'
        gg = myGraph(graphP=g, neighborIdx=g.neighbors, communityPartitions=comIdx, adjacencyMatrix=None)
        print('Calculating p, q ...')
        df['p'], df['q'] = communityConnectionProbabilities(gg)
        return gg, df




def df_empty(columns, dtypes, index=None):
    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df




def findLargerDivisible(num, numDiv):
    while (num % numDiv != 0): # math.remainder(num, numDiv)
        num += 1
    return num


class myGraph():
    def __init__(self, graphP, neighborIdx, communityPartitions, adjacencyMatrix):
        self.graphP = graphP
        self.neighborIdx = neighborIdx
        self.communityPartitions = communityPartitions
        self.adjacencyMatrix = adjacencyMatrix

