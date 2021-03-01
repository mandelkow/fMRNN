#!/usr/bin/env python
# coding: utf-8

# h_DsRnn_def.ipynb
# 
# ## Imports for h_DsRnn_v*.ipynb
# 
# SEE ALSO: `h_DsRnn_def_v1b.ipynb`
# 
# PREC: `h_DsRnn_v3b8.ipynb`, `h_DsRnn_def_v1b.ipynb`
# 
# AUTH: Hendrik.Mandelkow@nih.gov

# -----------------------------------------------------------------------------------------
# #### Biowulf preliminaries:
# ```bash
# module load python/3.6
# ...OR
# conda_on
# conda activate Tf1
# ```
# 
# ##### NB: GPU is required for TF backend!
# ```bash
# freengpu # free nodes
# sinteractive -TT -t 24:00:00 -c 14 --mem=16g --gres=gpu:p100:1,lscratch:32'
# alias sintg='f(){ sinteractive -TT -t 36:00:00 -c $((14*${1##*:})) --mem=64g --gres=gpu:$1,lscratch:10; }; f'
# sintg 1
# sintg p100:1
# 
# sjobs
# newwall --jobid ??? --time 36:00:00
# ```

# ## Tensorboard
# ```python
# %load_ext tensorboard
# %tensorboard --logdir ./TB_logs --port $PORT2
# ```

# # Imports

# WARNING: Guess this must come before imports!?!
# HOWTO enable autoreload for imported modules.
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
# %matplotlib auto
get_ipython().run_line_magic('matplotlib', 'inline')
#%aimport ...


## IMPORTS
import sys, os, re, glob
import numpy as np
np.set_printoptions(3)
get_ipython().run_line_magic('precision', '3')
# import scipy
import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
#plt.style.use('default')
plt.style.use('dark_background') # 'default', 'dark_background', 'seaborn-talk'
# plt.style.use('seaborn-talk')
from pprint import pprint
import scipy
from scipy import signal as scsi
import hdf5storage as hdf5
import nibabel as niba
import nilearn as nile
import nilearn.plotting as niplt
# HOWTO suppress warnings:
from warnings import warn, filterwarnings, simplefilter
filterwarnings('ignore',module='nilearn',lineno=1569)
# simplefilter('ignore')


# ### import htools

if '/hpy:' not in ':'.join(sys.path)+':':
    sys.path.insert(0,'/home/mandelkowhc/matlab/htools1/hpy')
# pprint(sys.path[:5])
os.chdir('/home/mandelkowhc/matlab/htools1/hpy')
get_ipython().system('jupyter nbconvert --to python --TemplateExporter.exclude_output=True --TemplateExporter.exclude_raw=True --TemplateExporter.exclude_markdown=False htools_v1b.ipynb')

from htools_v1b import hsys, hcd, hstd, hmovmean, hhline, hvline, hscalez, hreshape,     hFpath, hFname, htcode64, htime64, himgtileax, hxcorry, hnormalize, hrescale,     hsavefig, ddict, hplotchan
# from htools_v1b import hipymagic, hipyshell
from htools_v1b import hformatwarnmsg
import warnings
warnings._formatwarnmsg_impl = hformatwarnmsg

import hbiopack as hbp
import hDsCl as hds
# from h_BpRnnBw_def import *


hrms = lambda X,d: np.sqrt(np.mean(np.abs(X)**2,d))
hrss = lambda X,d: np.sqrt(np.sum(np.abs(X)**2,d))
hnorm2 = lambda X,d: np.sqrt(np.sum(np.abs(X)**2,d))
hcorrxy = lambda x,y: np.corrcoef(x,y,rowvar=False)[:x.shape[1],x.shape[1]:]


hplotstyles = lambda : [ plt.style.use('dark_background'), mpl.rcParams.update({'figure.figsize': (18,4)}) ];
hplotstyles = lambda : [ plt.style.use('seaborn-talk'), mpl.rcParams.update({'figure.figsize': (18,4)}) ];


hind2sub = np.unravel_index # lin.index to multi-subscripts
hsub2ind = np.ravel_multi_index # to lin.index

def hPutMaskSorted(M,X,x=0):
    '''[***] Similar to np.putmask but with sorted M.values
    '''
    # np.put( np.zeros(M.shape, X.dtype), np.unravel_index(np.argsort(M,None)[-X.size:],M.shape), X )
    # Img = np.zeros(list(M.shape)+X.shape[1:], X.dtype) + x
    # Img = np.zeros(M.shape, X.dtype) + x
    Img = np.full(M.shape, x, dtype=X.dtype)
    # [+++] HOWTO assign vlues to an ordered mask:
    Img[np.unravel_index(np.argsort(M,None)[-X.size:],M.shape)] = X
    return Img

# Use different name + input sequence?
hma2im = lambda X,M,x=0: hPutMaskSorted(M,X,x)

# Can't work bc np.put returns None!?
# [+--] hPutSortedMask = lambda Y,M,X: np.put( Y=np.zeros(M.shape, X.dtype), np.unravel_index(np.argsort(M,None)[-X.size:],M.shape), X )


get_ipython().run_cell_magic('script', '_bash', 'cd $ExId.results\nset +e # don\'t exit on error\nrm -f McPar.1D\nln -s dfile.r01.1D McPar.1D || echo Link exists.\n3dAFNItoNIFTI -overwrite -prefix Epi_mask.nii.gz full_mask.*.BRIK*\nif $OW || [ ! -e Epi_Mc.nii* ] ; then\n\t3dAFNItoNIFTI -overwrite -float -prefix Epi_Mc.nii.gz pb01.*.BRIK*\nfi\nif $OW || [ ! -e Epi_Mcr.nii* ] ; then\n\t3dAFNItoNIFTI -overwrite -float -prefix Epi_Mcr.nii.gz errts.*.BRIK*\nfi\nif $OW || [ ! -e Epi_Mcr_std.nii* ] ; then\n\t3dTstat -overwrite -stdev -prefix Epi_Mcr_std.nii.gz Epi_Mcr.nii*\nfi\nif $OW || [ ! -e Epi_Mc_mean.nii* ] ; then\n\t# 3dTstat -overwrite -mean -std -prefix Epi_Mc_mean+std.nii.gz Epi_Mc.nii*\n\t3dTstat -overwrite -mean -prefix Epi_Mc_mean.nii.gz Epi_Mc.nii*\n\t3dTstat -overwrite -stdev -prefix Epi_Mc_std.nii.gz Epi_Mc.nii*\nfi\n# find . -iname "*mask*.BRIK" -print -exec 3dAFNItoNIFTI {} \\;')


MASKVAL = +0.0


# ******************************************************************
# # Batch generator
# Training a *stateful* RNN on multiple (independent) voxels in parallel requires specially formed batches of training data. These could either be created and fed manually using model.train_on_batch() or by using model.fit_generator() with a custom generator - see below.
# 
# #### Re: Seq. length
# In simple Keras the seq. length is the same for training and prediction. It determines e.g. the "depth" of backprop in time. If the RNN *stateful* stride should equal seq. length and the batch size Nbatch equal to 1 or, perhaps, the number of inputs (voxels) trained in parallel. If the network is *not* stateful prediction may require longer sequences, but they can overlap i.e. the *stride* can be short e.g. 1TR.
# 

from scipy.sparse import csr_matrix

def hXtv2Data( X, Y, dt, t0=None, MaskVal=MASKVAL, MaskCol=False, Sparse=False):
    '''[***3ab] Cat time series of unequal sampling rate.
    Cat X with Y upsampled by dt with t0 offset and MaskVal between samples.
    
    USE: Data[:,:NX+NY] = hXtv2Data( Bp.Fata[:,:NX], Xtv[:,:NY], Bp.Fs*TR )
    
    RETURNS: Data with Data[:,:NX] = X[:,NX] and Data[ t0::dt, NX:] = Y[:,:NY]
    
    MaskVal: (scalar float) used to interpolate Y
    MaskCol: if True prepend Y with a "boolean" input mask column like:
        Y = np.c_[ Y[:,0]*0+1, Y ] and NX += 1
    Sparse: if True return (mem.efficient) sparse type csr_matrix
    '''
    
    '''
    if MaskCol: # Prepend a mask column of 0/1
        Y = np.c_[ Y[:,0]*0+1, Y ]
        # Y = np.c_[ np.ones_like(Y[:,0]), Y ]
    '''
    
    NX,NY = X.shape[1],Y.shape[1]
    if t0 is None:
        t0 = dt-1

    Nt = min( Y.shape[0], X.shape[0]//dt)
    if Sparse:
        # CSR: Sparse matrix stored in contiguous rows:
        Data = csr_matrix( (Nt*dt+t0, NX+NY), np.float32)
    else:
        Data = np.zeros( (Nt*dt, NX+NY), np.float32)

    if MaskVal:
        Data[...] = MaskVal # Use masking value for Xtv
        
    # TEST: print(X.shape,end=' '); print(Y.shape,end=' '); print(Data.shape)
    Data[:,:NX] = X[:Data.shape[0],:] # +++
    # Data[t0:Y.shape[0]*dt:dt, NX:] = Y[:,:NY]
    Data[t0::dt, NX:] = Y[:,:NY]
    
    if MaskCol:
        # numpy.insert(arr, obj, values, axis=None)
        Data = np.insert( Data, NX, Data[:,-1]!=MaskVal, axis=1)
    
    return Data


# ## v4b new ValFrac
# See h_DsRnn_def_v1b.ipynb for older / alternate versions.
# 

# ### TODO:
#  * [x] add Tsh for augmentation
#  * [ ] add suffle + reverse as control
#  * possibly avoid cp of data by reshaping batch instead of data?
#  * Incorporate upsampling of MRI data?
#  * set shift = 0 to test training

# TODO: Add data augmentation by random time shift.
class hBatchSeq1y(keras.utils.Sequence):
    '''[**4b++] Batch generator (keras.*.Sequence) for stateful RNN with NX,NY,NB > 1.
    Stack NB and NY into a batch 
    For use with STATEFUL=True
    Returns: ( X[NY*NB,NT,NX+1], Y[NY*NB,NT,1]) or ( X, Y, Mask )
    
    Data[t,:NX+NY]
    NT: length of each training sequence [ batch_size=(NY*NB, NT, NX+1) ]
    NB: split Data[t,c] into NB sections along t for parallel training
    NY= Data.shape[-1]-NX : split Data[t,:NX+NY] for parallel training of each NY
    Drop: Dropout 0 < Drop < 1.0, mask Drop*100% of input Y at random [DropMode= 'sample']
        DropMode='sequence' # drop Y input for entire samples (sequences - NT) at random
        DropMode='odd' # drop Y input for odd samples (sequences - NT)
        DropMode='last' # drop Y input for last samples (sequences - NT) in each section
    Mask = *sample_weights* = either 1D array of Batch.shape[0] or 2D of Batch.shape[:2]
        Return ( X, Y, sample_weights) to serve as a mask for cost functions
        Need to set sample_weight_mode = 'temporal' ?!?
    ValFrac: if >0 leave out ValFrac*100% at the end of each section
    WARNING: This results in *distributed* val.data e.g. NB=4, ValFrac=1/3 -> TTV,TTV,TTV,TTV
    ValFrac > 0 : deliver training data
    ValFrac < 0 : deliver validation data
    
    .getX() : retrieve X (full TS)
    .getY() : retrieve Y for comparison with Yh
    .predict( model ) : compute Yh
    .evaluate( model ) : compute losses
    .reshapeInput : change model input_shape for prediction on different batch size
    ...
    
    SEE: h_BpRnnBw_def.py
    '''
    # AUTH: Hendrik.Mandelkow@gmail.com
    
    def __init__(self, Data, NT, NX=None, NB=1, ValFrac=0, Mask=None, Drop=0, DropMode='sample', Xtrafo=None):
        # self.__dict__.update(Data=Data, NT=NT, NY=NY, NB=NB, NX=Data.shape[-1]-NY)
        self.__dict__.update(NT=NT, NX=NX, NB=NB, Mask=Mask, Drop=Drop, DropMode=DropMode, MaskVal=-10)
        self.__dict__.update(Xtrafo=Xtrafo)
        try: self.Drop, self.DropMode = self.Drop[0], self.Drop[1]
        except: pass
        self.MaskVal = +0.0; warn('+++ TEST +++ MaskVal.')
        if NX is None: self.NX = NX = Data.shape[-1]-1
        self.NY = Data.shape[-1] - self.NX
        self.Xlead = self.RandXlead = 0
        assert not (self.RandXlead or self.Xlead and ValFrac), 'Oops! RandXlead and ValFrac are likely incompatible.'
        
        #< Data[:,self.NX:] = hzscore(Data[:,self.NX:])
        
        assert Data.ndim <= 3, 'Oops!'
        if False and Data.ndim>2:
            warn('Using (NB,NT,NXY) = Data.shape')
            self.NT = NT = Data.shape[-2]
            self.NB = NB = Data.shape[0]
            Data = Data.reshape(-1,Data.shape[-1])
            
        ### Transform X input - very BETA
        if isinstance( self.Xtrafo, str): self.Xtrafo = [ self.Xtrafo ]
        if self.Xtrafo[0]=='step':
            # import pdb; pdb.set_trace()
            n = Data[:,-1] != self.MaskVal
            tmp = Data[ n, self.NX: ]
            tmp = np.diff(tmp,axis=0,prepend=0)
            Data[ n, self.NX:] = tmp
            Data[ :, self.NX:] = np.cumsum( Data[ :, self.NX:], 0)

        ### Reshape Data
        Data = Data[:Data.shape[0]//NT//NB*NB*NT,:].reshape(NB,-1,NT,Data.shape[-1]) # [NB,B,NT,NX+NY]
        Data = np.moveaxis(Data,0,1) # [B,NB,NT,NX+NY]
        if ValFrac:
            print('+++ WARNING: *Interleaved* validation data at the end of each block.')
        if ValFrac > 0:
            print('+ Training data.')
            Data = Data[:-round(abs(ValFrac)*Data.shape[0])]
        elif ValFrac < 0:
            print('+ Validation data.')
            Data = Data[-round(abs(ValFrac)*Data.shape[0]):]

        print('Batches per epoch: %u, batch size (NY*NB): %u'               %( Data.shape[0], Data.shape[1]*self.NY))
        
        assert Data.size > 0, 'Oops, Data.shape= '+str(Data.shape)
        
        self.Data = Data

    def __len__(self):
        return self.Data.shape[0]

    def __getitem__(self,idx): # return one *batch*!
        NT,NX,NY,NB = map(self.__dict__.get, ['NT','NX','NY','NB'])
        X = self.Data[idx] # One batch: X[NB,NT,NX+NY]
        if True:
            X = np.concatenate([ X[:,:,[*range(NX),NX+y]] for y in range(NY)], axis=0) # X[NY*NB,NT,NX+1]
        else:
            assert False,'NOT TESTED!'
            warn('NOT TESTED!')
            X = np.concatenate((np.tile(X[:,:,:NX],[1,1,1,NY]),X[:,:,None,:]),2) # [NB,NT,NX+1,NY]
            X = np.moveaxis(X,-1,0).reshape(NY*NB,NT,NX+1) # X[NY*NB,NT,NX+1]

        ### Shift Y in time to make prediction non-trivial.
        # Circshift each seq. (NT) may be suboptimal but simple and irrelevant.
        Y = np.roll(X[:,:,-1:],-1,axis=-2) # +++ Y[NY*NB,NT,1] (out) shifted -1 rel to X
        # Y = np.roll(X[:,:,-1:],0,axis=1); warn('TEST TEST TEST!')

        ### Transform X input
        if isinstance( self.Xtrafo, str): self.Xtrafo = [ self.Xtrafo ]
        if not self.Xtrafo:
            pass
        elif self.Xtrafo[0]=='step': # restore mask values in Y
            # if np.all( self.Data[ self.Data[:,NX+1]!=1, NX+1] == 0): # boolean mask col. NX+1
            tmp = np.diff( Y, axis=-2, prepend=Y[...,:1,:]) == 0
            Y[ tmp ] = self.MaskVal
            Y[:,-1,-1] = self.MaskVal
        elif self.Xtrafo[0]=='randn':
            X[...,self.Xtrafo[1]] = np.random.randn(**X[...,self.Xtrafo[1]].shape)
        elif self.Xtrafo[0]=='zeros':
            X[...,self.Xtrafo[1]] = 0
        elif self.Xtrafo[0]=='const':
            X[...,self.Xtrafo[1]] = self.Xtrafo[2]
        else:
            assert (self.Xtrafo is None), 'Oops! Parameter Xtrafo not recognized.'
        
        ### Dropout to decrease reliance on BOLD autocorrelations
        if self.Drop:
            assert (0 <= self.Drop <= 1), 'Oops! Expecting 0 < Drop < 1.'
            tmp = self.DropMode[:3].lower()
            if tmp in ['sam']: # samples
                assert not self.Xtrafo, 'Error, combined Xtrafo and Drop Samples is not implemented.'
                X[ np.random.random(X.shape[:-1])<self.Drop, -1] = self.MaskVal # ***
            if tmp in ['seq']: # sequences
                X[ np.random.random(X.shape[:1])<self.Drop, :, -1] = self.MaskVal # ***
            if tmp in ['odd']: # odd sequences 1,3,5,...
                if (idx % 2): X[ :, :, -1] = self.MaskVal # ***
            if tmp in ['las']: # drop last len()*Drop sequences 
                if idx >= len(self)*(1-self.Drop): X[ :, :, -1] = self.MaskVal # ***
            if np.all( np.logical_or( X[...,-2]==0, X[...,-2]==1) ):
                X[...,-2] = np.logical_and( X[...,-2], X[...,-1]!=self.MaskVal)
        
        ### Return mask?
        #< X[X==np.nan] = self.MaskVal
        #< X[ np.isnan(X[...,-1]), -1] = self.MaskVal
        #< assert not np.any(np.isnan(X)), 'Oops NaNs in X!?!'
        if self.Mask is None:
            return ( X, Y )
        elif isinstance( self.Mask, np.ndarray ):
            # assert False, 'Not tested!?!'
            return ( X, Y, self.Mask )
            # Could use Mask = np.any(Y,-1).astype(float)
        else:
            assert False, 'TEST! Not this way!'
            Mask = np.any( Y != self.Mask, -1).astype(float) # Mask[NY*NB,NT] might be correct?!?
            #< Mask = np.any( np.logical_not(np.isnan(Y)), -1).astype(float) # Mask[NY*NB,NT] might be correct?!?
            return ( X, Y, Mask ) # (..., sample_weights)
        """
        try:
            assert False, 'Not tested!?!'
            assert self.Mask.size > 1, 'Not an array!?'
            return ( X, Y, self.Mask )
            # Could use Mask = np.any(Y,-1).astype(float)
        except (AttributeError): # AssertionError
            # assert False, 'Error: Work in progress.'
            Mask = np.any( Y != self.Mask, 2).astype(float) # Mask[NY*NB,NT] might be correct?!?
            return ( X, Y, Mask )
        """
        
    def on_epoch_end(self):
        '''TODO: Add some data augmentation eg by random t-shift here.
        '''
        # It should really only matter for card (and resp)?
        if self.RandXlead:
            self.setXlead( np.random.randint( 1+self.RandXlead ) ) # new random shift
    
    
    def setXlead(self, Xlead, Ch=slice(2) ):
        '''Tshift Xchans left to lead Y by Xlead samples.
        : self.Xlead= 0: Current (random) time shift (left) in samples np.roll(Data[:,Ch],-Xlead,0)
        : self.RandXlead= 0: Max for random self.Xlead
        self.setXlead(0) : undo random time shift
        '''
        # assert False, 'Oops! setRandXlead() is untested.'
        # It should really only matter for card (and resp)?

        ## self.getX, transform, reshape and store as self.Data
        NT,NX,NY,NB = map(self.__dict__.get, ['NT','NX','NY','NB'])
        Data = self.Data # [B,NB,NT,NX+NY]
        Data = np.moveaxis(Data,0,1) # [NB,B,NT,NX+NY]
        Data = Data.reshape( -1, Data.shape[-1]) # [T,NX+NY]
        
        #< if self.Xlead is None: slef.Xlead = 0 # unnecessary
        Data[:,Ch] = np.roll( Data[:,Ch], self.Xlead-Xlead, 0) # apply new shift (left)
        self.Xlead = Xlead
        
        #< Data = Data[:Data.shape[0]//NT//NB*NB*NT,:].reshape(NB,-1,NT,Data.shape[-1]) # [NB,B,NT,NX+NY]
        Data = Data.reshape(NB,-1,NT,Data.shape[-1]) # [NB,B,NT,NX+NY]
        Data = np.moveaxis(Data,0,1) # [B,NB,NT,NX+NY]
        self.Data = Data
        
        # return # what?
    
    
    def getX(Bgen):
        '''[*1a+]
        '''
        # PREC: hBatchGen_getX()
        # OK this works, according to the test below.
        XY = 0 # XY = 0,1 = getX, getY
        NT,NX,NY,NB = map(Bgen.__dict__.get, ['NT','NX','NY','NB'])
        x = np.stack([ Bgen[n][0] for n in range(len(Bgen)) ],1) # y[NY*NB,B,NT,NX+1]
        assert NT == x.shape[-2], 'Oops!'
        assert NB == x.shape[0]//NY, 'Oops!'
        assert (NX+1) == x.shape[-1], 'Oops!'
        x = x.reshape(NY,NB,len(Bgen),NT,NX+1) # [NY,NB,B,NT,NX+1]
        x = np.concatenate((x[0,:,:,:,:NX], np.moveaxis(x[:,:,:,:,-1],0,-1)),-1) # [NB,B(PE),NT,NX+NY]
        x = x.reshape(-1,x.shape[-1]) 
        return x

    def getY(Bgen,Tsh=False):
        '''[*1a+]
        Tsh=True : Undo t-shift for training.
        Tsh=False : directly comparable to Yh
        NB : batch size, nof samples per batch
        B : nof batches per epoch
        '''
        XY = 1 # get Y
        NT,NX,NY,NB = map(Bgen.__dict__.get, ['NT','NX','NY','NB'])
        y = np.stack([ Bgen[n][XY] for n in range(len(Bgen)) ],1) # y[NY*NB,B,NT,1]
        assert NT == y.shape[-2], 'Oops!'
        assert NB == y.shape[0]//NY, 'Oops!'
        y = np.reshape(y,(NY,NB,len(Bgen),NT)) # [NY,NB,B,NT]
        # y = np.transpose(y,(1,2,3,0)) # y[NB,B,NT,NY]
        y = np.moveaxis(y,0,-1) # y[NB,B,NT,NY]
        if Tsh:
            y = np.roll(y,1,-2)
        y = y.reshape(-1,y.shape[-1]) # y[NB*N*NT,NY]
        return y

    def unbatchYh(self,Yh,NY=None,NB=None,Tsh=False):
        # This could be a static function Yh2Y()?
        '''
        # Yh[B*NY*NB,NT,1]
        Yh[t,NY] = hUnbatchYh( RNN.predict_generator( hBatchSeq1y(...)))
        '''
        NY = self.NY if NY is None else NY
        NB = self.NB if NB is None else NB
        NT = Yh.shape[-2]
        yh = np.reshape(Yh,(-1,NY,NB,NT)) # yh[B,NY,NB,NT]
        yh = np.transpose(yh,(2,0,3,1)) #yh[NB,B,NT,NY]
        if Tsh:
            yh = np.roll(yh,1,-2)
        yh = yh.reshape(-1,yh.shape[-1]) # yh[NB*B*NT,NY]
        return yh
    
    def predict(Bgen, RNN, Tsh=False, Reset=True):
        '''Run generator batches through RNN and reshuffle output into an array.
        Bgen = TrainGen or ValidGen
        '''
        if Reset:
            RNN.reset_states()
        Yh = RNN.predict_generator(Bgen,verbose=1) # Yh[B*NY*NB,NT,1]
        #< print(Yh.shape)
        # Yh = hUnbatchYh(Yh,Bgen.NY,Bgen.NB,Tsh)
        Yh = Bgen.unbatchYh(Yh,Bgen.NY,Bgen.NB,Tsh)
        return Yh

    def predict1(Bgen, RNN, Tsh=False, Reset=True):
        '''Predict using new RNN with input shape matching Bgen.
        E.g. use NB=1 for better stateful prediction.
        Bgen = TrainGen or ValidGen
        '''
        # RNNp = keras.models.clone_model(RNN)
        RNNp = keras.models.model_from_json(RNN.to_json())
        # RNNp._layers[1].batch_input_shape = (NY,NT,NX+1)
        RNNp._layers[1].batch_input_shape = Bgen[0][0].shape
        RNNp = keras.models.model_from_json(RNNp.to_json())
        RNNp.set_weights(RNN.get_weights())
        # [ RNNp.layers[n].set_weights(RNN.layers[n].get_weights()) for n in range(len(RNN.layers))]
        # ??? [ L.stateful= True for L in RNNp.layers if hasattr(L,'stateful') ]
        # ??? [ L.batch_input_shape= (1,None,3) for L in RNNp.layers if hasattr(L,'stateful') ]
        RNNp.summary()

        if Reset:
            RNNp.reset_states()
        Yh = RNNp.predict_generator(Bgen,verbose=1) # Yh[B*NY*NB,NT,1]
        #< print(Yh.shape)
        # Yh = hUnbatchYh(Yh,Bgen.NY,Bgen.NB,Tsh)
        Yh = Bgen.unbatchYh(Yh,Bgen.NY,Bgen.NB,Tsh)
        return Yh

    def reshapeInput(Bgen, RNN, InputShape=None):
        '''Cp RNN weights to new model with batch_input_shape matching generator (for prediction)
        Bgen = TrainGen or ValidGen
        RNN
        InputShape = batch_input_shape = (NB,NT,NY)
        '''
        # K.clear_session()
        # RNNp = keras.models.clone_model(RNN)
        RNNp = keras.models.model_from_json(RNN.to_json())
        # RNNp._layers[1].batch_input_shape = (NY,NT,NX+1)
        if InputShape is None: InputShape =  Bgen[0][0].shape
        # RNNp._layers[1].batch_input_shape = Bgen[0][0].shape
        RNNp._layers[1].batch_input_shape = InputShape
        RNNp = keras.models.model_from_json(RNNp.to_json())
        RNNp.set_weights(RNN.get_weights())
        # [ RNNp.layers[n].set_weights(RNN.layers[n].get_weights()) for n in range(len(RNN.layers))]
        # ??? [ L.stateful= True for L in RNNp.layers if hasattr(L,'stateful') ]
        # ??? [ L.batch_input_shape= (1,None,3) for L in RNNp.layers if hasattr(L,'stateful') ]
        
        try:
            RNNp.compile(optimizer=RNN.optimizer, loss=RNN.loss, metrics=RNN.metrics)
        except:
            warn('Compile failed?!?')
            pass
        
        RNNp.summary()

        return RNNp

    def evaluate(Bgen, RNN, Reset=True):
        '''
        Bgen = TrainGen or ValidGen
        '''
        if Reset:
            RNN.reset_states()
        Losses = RNN.evaluate_generator(Bgen,verbose=1) # Yh[B*NY*NB,NT,1]
        Losses = dict( zip( RNN.metrics_names, Losses))
        return Losses


def hUnbatchYh(Yh,NY,NB,Tsh=False):
    '''
    # Yh[B*NY*NB,NT,1]
    Yh[t,NY] = hUnbatchYh( RNN.predict_generator( hBatchSeq1y(...)))
    '''
    NT = Yh.shape[-2]
    yh = np.reshape(Yh,(-1,NY,NB,NT)) # yh[B,NY,NB,NT]
    yh = np.transpose(yh,(2,0,3,1)) #yh[NB,B,NT,NY]
    if Tsh:
        yh = np.roll(yh,1,-2)
    yh = yh.reshape(-1,yh.shape[-1]) # yh[NB*B*NT,NY]
    return yh


def hPredictGen(RNN, Bgen, Tsh=False, Reset=True):
    '''
    Bgen = TrainGen or ValidGen
    '''
    if Reset:
        RNN.reset_states()
    Yh = RNN.predict_generator(Bgen,verbose=1) # Yh[B*NY*NB,NT,1]
    print(Yh.shape)
    Yh = hUnbatchYh(Yh,Bgen.NY,Bgen.NB,Tsh)
    return Yh


def hBatchGen_getY(Bgen,Tsh=False):
    '''[*1a+]
    Tsh=True : Undo t-shift for training.
    Tsh=False : directly comparable to Yh
    NB : batch size, nof samples per batch
    B : nof batches per epoch
    '''
    XY = 1 # get Y
    NT = Bgen.NT
    NY = Bgen.NY
    NB = Bgen.NB
    y = np.stack([ Bgen[n][XY] for n in range(len(Bgen)) ],1) # y[NY*NB,B,NT,1]
    assert NT == y.shape[-2], 'Oops!'
    assert NB == y.shape[0]//NY, 'Oops!'
    y = np.reshape(y,(NY,NB,len(Bgen),NT)) # [NY,NB,B,NT]
    # y = np.transpose(y,(1,2,3,0)) # y[NB,B,NT,NY]
    y = np.moveaxis(y,0,-1) # y[NB,B,NT,NY]
    if Tsh:
        y = np.roll(y,1,-2)
    y = y.reshape(-1,y.shape[-1]) # y[NB*N*NT,NY]
    return y


def hBatchGen_getX(Bgen):
    '''[*1a+]
    '''
    # OK this works, according to the test below.
    XY = 0 # XY = 0,1 = getX, getY
    NT = Bgen.NT
    NX = Bgen.NX
    NY = Bgen.NY
    NB = Bgen.NB
    x = np.stack([ Bgen[n][0] for n in range(len(Bgen)) ],1) # y[NY*NB,B,NT,NX+1]
    assert NT == x.shape[-2], 'Oops!'
    assert NB == x.shape[0]//NY, 'Oops!'
    assert (NX+1) == x.shape[-1], 'Oops!'
    x = x.reshape(NY,NB,len(Bgen),NT,NX+1) # [NY,NB,B,NT,NX+1]
    x = np.concatenate((x[0,:,:,:,:NX], np.moveaxis(x[:,:,:,:,-1],0,-1)),-1)
    x = x.reshape(-1,x.shape[-1])
    return x


# # Loss Functions:
# Missing training data can be handled either by passing sample_weights [with sample_weight_mode='temporal'] for masking the loss function in model.fit(). Alternatively, we may define a custom loss function (with integrated mask) as seen below.
# 

# NB: Custom objects must be passed to model.compile and also model_load()
if 'KCustoms' not in locals(): KCustoms = {}


# NOTE: Looks like loss functions can return either a scalar or an array that will be summed over.
# I think sample_weight should require the array, but there is no error?!
# Maybe some erroneous broadcast goint on?!?

# https://github.com/keras-team/keras/blob/master/keras/losses.py
import keras.backend as K
def hMSE(Y, Yh):
    if not K.is_tensor(Yh): Yh = K.constant(Yh)
    Y = K.cast(Y, Yh.dtype)
    return K.mean(K.square(Yh - Y),-1) # axis=-1 didn't matter?!


# OK
import keras.backend as K
def hWMSE(Y,Yh):
    '''Weighted Mean Square Error (MSE)
    '''
    Mask= -10.0
    Mask= +0.0 # +++ TEST +++
    if not K.is_tensor(Yh): Yh = K.constant(Yh)
    Y = K.cast(Y, Yh.dtype)
    mask = K.not_equal(Y,Mask) # OK! could use NaN <-> 0
    # TODO: assert there are masked values?!
    # return K.mean(K.square(Yh[mask]-Y[mask])) # BUT: No boolean indexing in Keras!
    mask = K.cast(mask,K.dtype(Y)) # OK!
    return K.sum(K.square(K.abs(Yh - Y)*mask))/K.sum(mask) # OK!

    # return (K.square(K.abs(Yh - Y)*mask))/K.sum(mask)*K.size(Y) # Keras would apply mean implicitly?!
    # L = K.sum(K.square(K.abs(Yh - Y)*mask),-1)/K.sum(mask) # Keras would apply mean implicitly?!
    # return L * K.size(L)

KCustoms['hWMSE'] = hWMSE


import keras.backend as K
def hMSEmask(Mask=-10.0):
    def LossFun_(Y,Yh):
        '''Weighted Mean Square Error (MSE)
        '''
        if not K.is_tensor(Yh): Yh = K.constant(Yh)
        Y = K.cast(Y, Yh.dtype)
        mask = K.not_equal(Y,Mask) # OK! could use NaN <-> 0
        # return K.mean(K.square(Yh[mask]-Y[mask])) # BUT: No boolean indexing in Keras!
        mask = K.cast(mask,K.dtype(Y)) # OK!
        return K.sum(K.square(K.abs(Yh - Y)*mask))/K.sum(mask) # OK!
    return LossFun_

# KCustoms['hMSEmask'] = hMSEmask(-10.0)
warn('+++ TEST +++ Using alternate MaskVal!')
KCustoms['hMSEmask'] = hMSEmask(+0.0)
# hMSEmask = hMSEmask(+0.0) # This work better???
# KCustoms['hMSEmask'] = hMSEmask


import keras.backend as K
class hWMSEmaskCl: # Should be a sub-class of LossFunctionWrapper?!
    def __init__(self, Mask=-10.0):
        self.Mask = Mask
        
    def __call__(self,Y,Yh):
        '''Weighted Mean Square Error (MSE)
        '''
        if not K.is_tensor(Yh): Yh = K.constant(Yh)
        Y = K.cast(Y, Yh.dtype)
        mask = K.not_equal(Y,self.Mask) # OK! could use NaN <-> 0
        # return K.mean(K.square(Yh[mask]-Y[mask])) # BUT: No boolean indexing in Keras!
        mask = K.cast(mask,K.dtype(Y)) # OK!
        return K.sum(K.square(K.abs(Yh - Y)*mask))/K.sum(mask) # OK!

# KCustoms['hWMSE'] = hWMSEmaskCl()


import keras.backend as K
# NOTE: Y,Yh[NB,NT,NY] one batch!
# NOTE: If Y is not 0 mean, a constant prediction can yield hWRVF < 1.
def hWRVF(Y,Yh):
    '''Weighted Residual Variance Fraction
    '''
    Mask= -10.0
    Mask= +0.0 # +++ TEST +++
    if not K.is_tensor(Yh): Yh = K.constant(Yh) # need this?? or make variable?
    Y = K.cast(Y, Yh.dtype) # need this??
    mask = K.not_equal(Y,Mask) # OK! could use NaN <-> 0
    # return K.sum(K.square(Yh[mask]-Y[mask])) # BUT: No boolean indexing in Keras!
    mask = K.cast(mask,K.dtype(Y)) # OK!
    #<? Y = Y - K.sum( Y*mask, -2, keepdims=True)/K.sum( mask, -2, keepdims=True)
    # return K.sum(K.square((Yh - Y)*mask)) / K.sum(K.square(Y*mask))
    Loss = K.sum(K.square((Yh - Y)*mask)) / K.sum(K.square(Y*mask - K.sum(Y*mask)/K.sum(mask))) # more appropriate?!!
    # assert K.eval( K.all( K.not_equal( Loss, np.nan))), 'Oops! NaN in output.' # TODO: make this work!
    # assert not np.any( np.isnan( K.eval( Loss))), 'Oops! NaN in output.' # TODO: make this work!
    return Loss

KCustoms['hWRVF'] = hWRVF


import keras.backend as K
def hRVF(Y,Yh):
    '''Residual Variance Fraction
    return K.sum(K.square(Yh - Y)) / K.sum(K.square(Y))
    '''
    if not K.is_tensor(Yh): Yh = K.constant(Yh)
    Y = K.cast(Y, Yh.dtype)
    return K.sum(K.square(Yh - Y)) / K.sum(K.square(Y))

KCustoms['hRVF'] = hRVF


# ------------------------------------------------------

#%% Correlation Loss
''' NOTE: Keras defines losses with additional inputs e.g. cosine_similarity in 3 steps: 
1) The actual loss fun [cosine_similarity(Y,Yh,axis=-1)] returns an array of losses for each batch of samples.
2) A wrapper sub-class( LossFunctionWrapper ), which is initialized with the extra parameters as well as
a "reduction" method usually sum over batch to yield a final scalar result.
https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/losses.py
'''
import keras.backend as K
def hCorrLossMask(Mask=0):
    def LossFun_(Y,Yh):
        '''
        With Mask==0 this is very close to keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
        '''
        if not K.is_tensor(Yh): Yh = K.constant(Yh) # Why?!
        Y = K.cast(Y, Yh.dtype)
        mask = K.not_equal(Y,Mask)
        # return K.mean(K.square(Yh[mask]-Y[mask])) # BUT: No boolean indexing in Keras!
        mask = K.cast(mask,K.dtype(Y)) # OK!
        
        # Yh[NB,NT,NY] ?!
        # Y = K.normalize( Y, -2) # should be scaled appropriately before ?!
        # Yh = K.normalize( Yh , -2)
        Yh = K.normalize( Yh * mask, -2)
        L = -K.sum( Y * Yh * mask, -2)
        # Apparently, by default Keras will take the mean, if the result of loss fun is not scalar.
        # In order for sample_weights to work Keras would need the array, wouldn't it?!
        # Might not even want to sum over time steps for that?!
        return L
    return LossFun_

KCustoms['hCorrLoss'] = hCorrLossMask(+0.0)


# # RNN model

# TODO: Consider activation functions?!
# AUTH: HM, 2019-06-26, v3b2: Add GRU, better handling of input layer.
# AUTH: HM, 2019-06-26, v3b3: Add Nio=Type
# def mkRNN(Nio=[1,1], Nsteps=1, Nbatch=None, Lpar, Cpar):
def mkRNN(Nio=[1,1], Nsteps=1, Nbatch=None, **kwarg):
    '''[++-] Make simple stateful RNN with final dense layer.
    Nsteps: number of time steps in each sample sequence
    Nbatch: training batch size (number of sample sequences per batch)
    TODO: What about initialization / regularization?
    '''
    PARS = {} # kwarg for Keras.layer...
    # batch_input_shape = (batchsize,timesteps,data_dim) only for input layer
    PARS['batch_input_shape'] = (Nbatch, Nsteps, Nio[0])
    PARS['return_sequences']=True
    PARS['stateful']=True # ***
    # PARS['activation'] = 'linear' # BAD!
    PARS.update(kwarg)
    
    RNN = keras.models.Sequential() # +++
    # RNN.name = '' # clear generic name sequential_1
    # The latest Keras (2.2.4?) does not seem to like name=''.
    # Also, the name parameter seems to have some obscure internal uses. Better not touch.
    
    Type = 'L' # *** default
    # RNN.name = str(Nio[0])+'L%u'*(len(Nio)-1)%tuple(Nio[1:])
    # RNN.name = '%uL%u'%tuple(Nio[:2])
    #< for Nout in Nio[1:]:
    for n,Nout in enumerate(Nio[1:]):
        if n==len(Nio)-2:
            PARS['activation'] = 'linear' # output layer w/ lin. activation
        if Nout in ['linear','tanh']:
            PARS['activation'] = Nout
            continue
        if isinstance(Nout,str):
            Type = Nout
            continue
        if Type == 'L':
            RNN.add(keras.layers.LSTM( Nout, **PARS))
        elif Type == 'G':
            # https://keras.io/layers/recurrent/#GRU
            # keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid',...
            RNN.add(keras.layers.GRU( Nout, implementation=2, **PARS))
        elif Type == 'D': # Dense
            RNN.add(keras.layers.Dense(1, activation='linear'))
        else:
            raise ValueError('Type must be L,G,D not: '+Type)
            
        #< RNN.name += Type+'%u'%Nout
            
    # tmp = [str([x.units for n,x in enumerate(y.layers)]) for y in RNNs]+['line %u'%x for x in range(n+2,len(h)+1)]
    PARS.pop('batch_input_shape',None) # remove key, only required for input layer

    # TODO: Use only hWMSE
    # TODO: Could use MaskVal; hWRVFmask(MaskVal) here
    # RNN.compile(loss=hWMSE, optimizer='adam', metrics=[hWRVF]) # +++
    RNN.compile(loss=KCustoms['hMSEmask'], optimizer='adam', metrics=[ KCustoms['hWRVF'] ]) # +++
    # RNN.compile(loss='MSE', optimizer='adam', metrics=[hWMSE,hWRVF], weighted_metrics=[hRVF,hMSE], sample_weight_mode='temporal') # TEST!!!
    # NOTE: Apparently, sample_weights are applied to "loss" as well as weighted_metrics.
    # TEST! RNN.compile(loss='MSE', optimizer='adam') # TEST!!!
    print('+ RNN.name= '+RNN.name)
    # Record input parameters:
    tmp = locals() # Dunno why this is necessary?!
    RNN.mkRNNargs = { n:tmp[n] for n in ['Nio','Nsteps','Nbatch'] }
    RNN.mkRNNargs.update(kwarg)
    return RNN


from keras.utils import multi_gpu_model

def mkRnnGpu(Model,**par):
    """[+++] Recompile model for multi-GPU training: MModel = multi_gpu_model(Model, **par)
    
    PAR = { # optional parameters / defaults
        'gpus':None, # Nof GPUs or None for all available
        'cpu_merge':True # ?!? force merging of weights on CPU
        'cpu_relocation':False, # ?!? force transfer of model from GPU to CPU
        }
        
    WARNING: Use Model not MModel for saving:
        model.save(Fname)
        model.save_weights(Fname)
        
    MModel.fit() will split batches (evenly) across GPUs.
    
    """

    try: # Train on mult. GPUs
        MModel = multi_gpu_model(Model, **par)
    except: # Train on 1 CPU or GPU
        MModel = Model

    MModel.compile( loss= Model.loss, optimizer= Model.optimizer, metrics= Model.metrics )
    
    return MModel


# # Callbacks

# ## hResetStatesCb callback

# TODO: Why reset on epoch end?
class hResetStatesCb(keras.callbacks.Callback):
    '''
    hResetStatesCb(False/True) # reset states at on_epoch_begin (default) / end
    '''
    def __init__(self,End=False): # required?
        self.End = End

    def on_epoch_begin(self, epoch, logs={}):
        if not self.End:
            # HOWTO access model from callback:
            self.model.reset_states()

    # Doppeltgemoppelt haelt besser - for val_loss also?!
    def on_epoch_end(self, epoch, logs={}):
        if self.End:
            # HOWTO access model from callback:
            self.model.reset_states()

# FITPAR['callbacks'] += [ hResetStatesCb() ]


# ## hTBoardTextCb: custom TensorBoard callback

# NOTE: Each TensorBoard CB will create a separate events file (for each run).
# To avoid duplicate logging of metrics etc., use only one TB callback in model.fit() 

# hTBoardTextCb v3b2: input {LogTag:LogStr,...} *OR* [(LogTag, LogStr),...]
from keras.callbacks import TensorBoard
# from keras.callbacks.tensorboard_v1 import TensorBoard # Get older version?!
import tensorflow as tf

hdict2list = lambda D: list( D.items() )

class hTBoardTextCb(TensorBoard):
    '''Tensorboard callback extended to log arbitrary text strings.
    USE:
    callbacks += [ hTBoardTextCb(log_dir, MyLogs, **kwargs)]
    with MyLogs = {'Tag1':'Text1',...} or [('Tag1','Text1'),...]
    
    NOTE: Each TBoard callback creates a separate events.* file (for each run).
    To avoid duplicate logging, use only one TB callback in model.fit()
    by replacing:
    callbacks += [keras.callbacks.TensorBoard(log_dir=TbDir,**kwargs)]
    
    Logs = [ ('Tag1', 'String1'), ('Tag2', 'String2'), ... ]
    Could use an OrderedDict (from collections). But a simple pythetic "dict" does not preserve order.
    
    **{'batch_size':NB, 'histogram_freq':0, 'write_graph':False, 'write_images':False}
    '''

    def __init__(self, log_dir, MyLogs=None, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.MyLogs = MyLogs

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.MyLogs is None:
            return
        try:
            MyLogs = list(self.MyLogs.items()) # convert dict to list
        except:
            MyLogs = self.MyLogs

        # Might consider: tf.summary.merge_all()
        for TagText in MyLogs:
            summary = tf.summary.text( TagText[0], tf.convert_to_tensor(TagText[1]) )
            # https://www.tensorflow.org/api_docs/python/tf/summary/text

            with  tf.Session() as sess:
                # No need for this?: self.writer = tf.summary.FileWriter('./Tensorboard', sess.graph)
                s = sess.run(summary)
                self.writer.add_summary(s)

        # Do we need sth like this?: self.writer.close()


# # hLoadKm

import concurrent.futures

def hLoadProcDir(ProcDir, Npool=None):
    '''
    metapar.mat, model.json, model_BesVal.h5, train_log.csv
    '''
    if not isinstance( ProcDir,str):
        with concurrent.futures.ProcessPoolExecutor(Npool) as pool: # pool size e.g. (8)?
            Pars = pool.map( hLoadProcDir, ProcDir)
        Pars = list(Pars)
        return Pars

    par = { 'ProcDir':ProcDir, 'Xtrafo':None }
    
    hdf5.loadmat( par['ProcDir']+'metapar.mat', par)
    par['ProcDir'] = ProcDir
    
    par['TrainLog'] = pd.read_csv( par['ProcDir']+'train_log.csv')
    
    hdf5.loadmat( par['ProcDir']+'EvalData.mat', par)
    par['ProcDir'] = ProcDir
    
    print('.',end='')
    
    return par

