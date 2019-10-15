***Work in progress. Not for distribution.***

# RNN models of the rs-fMRI signal
AUTHORS: H.Mandelkow <Mandelkow[at]icloud.com>

<!-- ## Synopsis -->
#### _Can DNNs be used to model the fMRI signal as a function of external stimuli and internal covariates like the heart beat and respiration?_
fMRI data analysis is traditionally treated as a regression problem and solved by fitting a *general linear model* (GLM) to individual voxel time series. A major disadvantage of the GLM approach lies in the fact that it depends on regressors with high explanatory power, which are hard to come by experimentally or based on intuition. DNNs might offer a systematic, data-driven solution to this problem. One straight-forward approach would be to replace the GLM with a nonlinear RNN model. RNNs are famous for modeling time series. Of course, the ideosyncrasies of neuroimaging data pose specific challenges including:

1. a limited amount of training data
2. data sampled on vastly different time scales
3. systematic heterogeneity that requires an adaptive ARX-type model to generalize across experiments, subjects and brain regions

Here we use Python3 + Keras/TF to develop RNN models that can be trained on undersampled (missing) data and learn to adapt (generalize) to variable input/output statistics.

## agenda:
- [x] simplified RNN model maker + custom loss
- [x] custom loss (masked MSE) to ignore missing data
- [x] custom batch generator for parallel GPU training of voxels and temporal sections
    - [x] generate missing data mask
    - [x] dropout random / systematic
    - [x] methods for validation: .predict .evaluate .getY .reshapeModel etc.

