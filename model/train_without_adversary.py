#train_without_adversary.py

import os
import sys
import time
import numpy as np
import models
import keras
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]

    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    X_full_train, X_sketch_train, X_full_val, X_sketch_val,target_train,target_val = data_utils.load_data(dset, image_data_format)
    img_dim = X_full_train.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

       

       

       # DCGAN_model = models.DCGAN(generator_model,
       #                           discriminator_model,
       #                            img_dim,
       #                            patch_size,
       #                            image_data_format)
        ##########################################################################                           
        classifier_model  =  models.Pereira_classifier(img_dim) 
        #classifier_model  = models.MyResNet18(img_dim) 
        #classifier_model  = models.MyDensNet121(img_dim)
        #classifier_model  = models.MyNASNetMobile(img_dim)
        
        
        
        
                                  
        #########################################################################
        loss = [keras.losses.categorical_crossentropy]
        loss_weights = [1]
        classifier_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        class_loss = 100
        disc_loss = 100
        max_accval =0
        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_full_batch, X_sketch_batch,Y_target in data_utils.gen_batch(X_full_train, X_sketch_train,target_train, batch_size):

               
                class_loss = classifier_model.train_on_batch(X_sketch_batch, Y_target)
                
                
                # Unfreeze the discriminator
                
                batch_counter += 1
                progbar.add( batch_size, values=[("class_loss", class_loss)])

                # Save images for visualization
                 
                    
                                                    

                if batch_counter >= n_batch_per_epoch:
                    X_full_batch, X_sketch_batch,Y_target_val = next(data_utils.gen_batch(X_full_val, X_sketch_val,target_val, int(X_sketch_val.shape[0])))  
                    y_pred  = classifier_model.predict(X_sketch_batch)
                    y_predd = np.argmax(y_pred,axis=1)
                    y_true  = np.argmax(Y_target_val,axis=1)
                    #print(y_true.shape)
                    accval=(sum((y_predd==y_true))/y_predd.shape[0]*100)
                    if (accval>max_accval):
                       max_accval = accval
                       
                    print('valacc=%.2f'% (accval))
                    print('max_accval=%.2f'% (max_accval))
                                    
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
    except KeyboardInterrupt:
        pass
