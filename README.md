# Interpretable 2-stage representation learning from raw waveform

The proposed model consists of a two-step relevance weighting approach over parametric layers. The first step performs relevance weighting on the output of the first layer of convolutions. This convolutional layer learns a parametric acoustic filterbank from the raw waveform. The cosine-modulated Gaussian kernels are used to design acoustic filterbank with learnable means.
The relevance weighted filterbank representation is used as input to the second convolutional layer which performs modulation filtering. This layer repeats the operations of the first layer in a 2-D fashion. 
The kernels of the second convolutional layer are 2-D spectro-temporal modulation filters (2-D cosine-modulated Gaussian kernels) with learnable rate-scale frequencies and the filtered representations are weighted using another relevance sub-network.
The full acoustic model consisting of relevance sub-networks, convolutional layers and feed-forward layers is trained for a speech recognition task.

******************************************************************

The script Net_raw_AcFB_Attn_ModFB_Attn_CNN2D_DNN_cuda.py contains the proposed network architecture. It takes the raw speech waveform in batches as input, each of size [B,1, 101, 400], for batch size B=32, t=101 raw frames and s=400 samples in each frame.

******************************************************************

Implementation of the paper:

P. Agrawal, S. Ganapathy, "Interpretable Representation Learning for Speech and Audio Signals Based on Relevance Weighting," IEEE Transactions and Audio, Speech and Language Processing, 2020.

******************************************************************

07-Sept-2020 See the file LICENSE for the licence associated with this software.
