# NPB-REC: Non-parametric Assessment of Uncertainty in Deep-learning-based MRI Reconstruction from Undersampled Data

## Abstract 

Uncertainty quantification in deep-learning (DL) based image reconstruction models is critical for reliable clinical decision making based on the reconstructed images.
We introduce ''NPB-REC'', a non-parametric fully Bayesian framework for uncertainty assessment in MRI reconstruction from undersampled ''k-space'' data. 
We use Stochastic gradient Langevin dynamics (SGLD) during the training phase to characterize the posterior distribution of the network weights. 
We demonstrated the added-value of our approach on the multi-coil brain MRI dataset, from the fastmri challenge, in comparison to the baseline E2E-VarNet with and without inference-time dropout. Our experiments show that NPB-REC outperforms the baseline by means of reconstruction accuracy (PSNR and SSIM of 34.55, 0.908 vs. 33.08, 0.897, p<0.01) in high acceleration rates (R=8). This is also measured in regions of clinical annotations. More significantly, it provides a more accurate estimate of the uncertainty that correlates with the reconstruction error, compared to the Monte-Carlo inference time Dropout method (Pearson correlation coefficient of R=0.94 vs. R=0.91). The proposed approach has the potential to facilitate safe utilization of DL based methods for MRI reconstruction from undersampled data. 
