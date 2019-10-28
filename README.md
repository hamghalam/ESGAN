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

# Prepare dataset

1- Put your dataset,here BraTS, on the root address:
2- Create "data_adr.txt" file and determine requirement as bellow:



# To train model, run this command on Linux terminal:

<div class="highlight highlight-source-shell"><pre>
python Enhancement_GAN.py config/data_adr.txt
</pre></div>



# How to download data
BraTS 2019 dataset. Data can be downloaded from http://braintumorsegmentation.org/
