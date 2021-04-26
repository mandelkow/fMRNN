
*Abstract ISMRM 2020*

# Recurrent Neural Networks reveal the effect of physiological factors on the global fMRI signal
**H. Mandelkow, D. Picchioni, J. de Zwart, J. Duyn**
1. *Advanced MRI section, LFMI, NINDS, National Institutes of Health, Bethesda, MD, USA*

### Main Findings
Recurrent neural networks (RNNs) have great potential for the analysis of resting-state fMRI data. They successfully predict non-linear interactions between the global fMRI signal and physiological covariates in human sleep.

## Synopsis
Recurrent neural networks (RNNs) are flexible and efficient data-driven models of multivariate time series. Given enough data to train such complex non-linear models, they could revolutionize the analysis of brain imaging data (fMRI) by replacing linear models of limited explanatory power. As a proof of concept and feasibility we trained RNNs successfully to model the global fMRI signal and its relationship with cardiac and respiratory signals during human sleep. The RNN model identified relevant features that are corroborated by recent findings obtained by traditional modeling approaches.

## Intro
BOLD fMRI has become the most important imaging modality in human neuroscience thanks to more than two decades of technological advances in the field. But the interpretation of fMRI experiments remains hampered by a conspicuous lack of accurate and comprehensive mathematical models that explain more than just a fraction of the fMRI signal variance and its complex relationship to intrinsic and extrinsic factors such as sensory stimuli. In recent years Deep Neural Networks (DNNs) have demonstrated remarkable potential as data-driven numerical models that are flexible, efficient and able to solve previously intractable problems. The application of DNNs to the complex modeling problem of fMRI analysis has obvious appeal, but the application to neuroscientific data is not straight-forward as a number of technical hurdles must be overcome. The biggest challenge lies in the fact that fitting DNN models with very large numbers of parameters typically requires equally large amounts of data that must moreover be representative in the sense that it samples all relevant aspects and variations of the underlying system to be modeled.

As a proof of concept and feasibility we present one of the first applications of a Recurrent Neural Network (RNN) model directly trained on resting-state fMRI data to predict the influence of simultaneously acquired physiological signals.

## Methods
A large set of resting-state fMRI data (82h) was acquired in 8 participants of an all-night sleep study that also included simultaneous measurements of the EEG, pulse-oximeter (PPG = photoplethysmograph) and a respiratory belt [1]. Imaging was performed on a 3T Siemens Skyra scanner using the 20-channel head coil and the following single-shot EPI sequence parameters: FA=90, TR = 3s, TE=36ms, resolution= 2.5x2.5x2mm +0.5mm gap, matrix= 96x70x50, GRAPPA=2x. PPG and respiratory signals were acquired using a finger probe and a chest belt (Biopac, Goleta, CA, USA). The fMRI data was pre-processed minimally using AFNI (afni.nimh.nih.gov) to perform only motion correction and regression of the resulting motion parameters. A whole-brain mask was applied and spatially averaged to obtain the global fMRI signal, which was furthermore detrended, standardized to unit variance and clipped at 3 standard deviations (SD) to remove outliers. The PPG and respiratory signals were similarly detrended, standardized and clipped after down-sampling the original signals from 1000Hz to 20Hz. All signal processing and RNN models were implemented in Python (3.6) using the SciPy and Keras/TensorFlow packages. An RNN with 4 input channels (PPG, resp, fMRI, fMRI mask) and one output channel connected by 3 layers of 64, 64 and 32 Gated Recurrent Units (GRU) [3] with tanh activation and one fully-connected unit with linear activation. The RNN training data combined fMRI and physiological signals in such a way that the network was trained to make a forward prediction of the fMRI signal, which lagged by one TR. By design the RNN delivered an output signal at the sampling rate of the input (20Hz), but a customized cost function (masked L2 norm) ensured that only the one available sample of fMRI data at the end of each TR contributed to the optimization procedure (ADAM algorithm). 

## Results+Discussion
The trained RNNs predicted a very significant fraction of the global fMRI signal, 30%-60% of the validation data variance on average, but the results on individual data segments varied widely (Figure 1). By training multiple RNNs on different input signals we identified the following dominant factors: 
1. Input signals: The pulse-ox. signal is shown to have by far the largest influence (predictive power) on the global fMRI signal. Closer inspection of the output waveform furthermore identifies the PPG amplitude as the determining factor (Figure 2). This is most interesting because it agrees well with a recently published analysis using human-engineered features in a more traditional linear regression model [2].
2. In the same vein, we note that the predictive power of RNN models was apparently sleep-stage dependent, because deep sleep is characterized by reduced variance in both the global fMRI signal and its physiological covariates. It is important to note that within a (long) fMRI experiment there may be regime changes that require a flexible model able to switch gears.
3. Finally, we acknowledge that the observed high prediction accuracies are contingent upon training and validation data taken from the same subjects and experiments. Cross-validation on left-out experiments, sessions and subjects yielded much higher residual variance fractions.

## Conclusions
RNN regression identified the PPG signal as most predictive of the global fMRI signal and revealed its amplitude to be the most relevant feature. These findings may well be specific to certain sleep stages represented in our data, but they are in line with recently published results and encourage us to conclude that RNNs have great potential for the data-driven (model free) analysis of resting-state fMRI and likely also task fMRI, if sufficiently large homogeneous data sets are available. Our RNNs were trained successfully and in a reasonable amount of time (2-4h on 1 GPU) using large but not exorbitant amounts of data (82h). The initial results presented here are all the more encouraging as little optimization has been done so far and there is obviously tremendous potential for gains in efficiency and analytical insight.

## Refereces
[1]: Moehlman et al. 2019. All-night functional magnetic resonance imaging sleep studies. J Neurosci Methods. 2019 Mar 15;316:83-98. doi: 10.1016/j.jneumeth.2018.09.019. Epub 2018 Sep 20.

[2]: Özbay et al. 2019. Sympathetic activity contributes to the fMRI signal. Communications Biology. in press

[3]: Cho et al. 2014. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". arXiv:1406.1078

## Figures
<img src="Fig/ISMRM2020_Fig1.png" alt="Fig.1" width=250 align=left>

***Figure 1 (left): Global fMRI signal variance explained by RNNs*** *trained on combinations of the input signals M,P,R (fMRI, pulse, respiration). Boxes show distributions across all sections of validation data.*
<!-- <br clear="both"/> -->
======================================================

***Figure 2 (below): Predicted global fMRI signal as a function of RNN input channels.***<br>
*RNNs were trained to make a 1-step-ahead prediction of the global fMRI signal (red dots) thus realizing a non-linear but causal filter of the pulse-ox. and respiratory input signals (grey lines). The colored lines show the output of 4 RNNs trained on combinations of the 3 available input channels: M,P,R (fMRI, pulse-ox., respiration). In the 2nd half of this 8-minute data segment the fMRI input was dropped to observe RNN predictions based on the physiological inputs alone. Thus we determined that the PPG signal present in models MPR (blue) and MP (green) contributes much more than the respiratory signal (orange).*

![Figure 2](Fig/ISMRM2020_Fig2.png)
