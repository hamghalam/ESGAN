# ESGAN
Enhancement and Segmentation GAN (ESGAN)
We present a novel architecture based on conditional generative adversarial networks (cGANs) to improve the lesion contrast for the pixel-wise segmentation. ESGAN effectively incorporates the classifier loss into the adversarial one during training to predict
the central labels of the sliding input patches.

You can find detailed results (Team name: Hamghalam) on BraTS 2013 dataset on:
<p> - Challenge Dataset - </p>
<p> https://www.smir.ch/BRATS/Start2013 </p>

![](https://github.com/hamghalam/ESGAN/blob/master/image.png)



# Prerequisites

<p> A CUDA compatable GPU with memory not less than 12GB is recommended for training. For testing only, a smaller GPU should be suitable. </p>
<p> Linux or OSX </p>
<p> NVIDIA GPU + CUDA CuDNN  </p> 
<p> Keras  </p>
<p> SimpleITK  </p>
<p> TensorFlow </p>


# Prepare Dataset
Put your Dataset as numpy array as:

<p> data shape (#samples, width, lenght, 1)  </p>

<p>  X_full       -------> High Contrast images based on FLAIR  </p>
<p>  X_sketch     -------> Original image FALIR                 </p>
<p>  Target_class -------> Segmentation labels (Ground trusth)  </p>


# To train model, run this command on Linux terminal:

<div class="highlight highlight-source-shell"><pre>
python main.py 16 16 --backend tensorflow --nb_epoch 100 --do_plot --generator deconv --n_batch_per_epoch 400
</pre></div>

<p>
  positional arguments:
    
    patch_size            Patch size for D

optional arguments:

    -h, --help            show this help message and exit
    --backend BACKEND     theano or tensorflow
    --generator GENERATOR
                        upsampling or deconv
    --dset DSET           facades
    --batch_size BATCH_SIZE
                        Batch size
    --n_batch_per_epoch N_BATCH_PER_EPOCH
                        Number of training epochs
    --nb_epoch NB_EPOCH   Number of batches per epoch
    --epoch EPOCH         Epoch at which weights were saved for evaluation
    --nb_classes NB_CLASSES
                        Number of classes
    --do_plot             Debugging plot
    --bn_mode BN_MODE     Batch norm mode
    --img_dim IMG_DIM     Image width == height
    --use_mbd             Whether to use minibatch discrimination
    --use_label_smoothing
                        Whether to smooth the positive labels when training D
    --label_flipping LABEL_FLIPPING
                        Probability (0 to 1.) to flip the labels when training
                        D
</p>



# How to download data
BraTS 2013 dataset. Data can be downloaded from http://braintumorsegmentation.org/
