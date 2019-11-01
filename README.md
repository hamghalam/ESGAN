# ESGAN
Enhancement and Segmentation GAN (ESGAN)
We present a novel architecture based on conditional generative adversarial networks (cGANs) to improve the lesion contrast for the pixel-wise segmentation. ESGAN effectively incorporates the classifier loss into the adversarial one during training to predict
the central labels of the sliding input patches.

You can find detailed results (Team name: Hamghalam) on BraTS 2013 dataset on:
<p> - Challenge Dataset - </p>
<p> https://www.smir.ch/BRATS/Start2013 </p>

![](https://github.com/hamghalam/ESGAN/blob/master/image.png)



![Alt Text](https://github.com/hamghalam/ESGAN/blob/master/Segmentation.gif)

<p> &#x1f4d2; Yellow : Edema    </p>
<p> &#x1F535; Blue   : Enhancing tumor </p>
<p> &#128215; Green  : Non-Enhancing tumor  </p>

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

<p>  X_full       -------> High contrast images based on FLAIR  </p>
<p>  X_sketch     -------> Original image FALIR                 </p>
<p>  Target_class -------> Segmentation labels (Ground trusth)  </p>


# To train model, run this command on Linux terminal:

<div class="highlight highlight-source-shell"><pre>
python main.py 16 16 --backend tensorflow --nb_epoch 100 --do_plot --generator deconv --n_batch_per_epoch 400
</pre></div>

<p>
  positional arguments:
    
    --patch_size            Patch size for D (here 16x16)
    --backend BACKEND       theano or tensorflow
    --generator GENERATOR   upsampling or deconv
    --n_batch_per_epoch     N_BATCH_PER_EPOCH  Number of training epochs
    --nb_epoch   NB_EPOCH   Number of batches per epoch
    
    --do_plot             Debugging plot
    
</p>


# How to download data
BraTS 2013 dataset. Data can be downloaded from http://braintumorsegmentation.org/
