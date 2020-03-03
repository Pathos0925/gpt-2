#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='345M'    
):
    """
    Interactively run the model
    :model_name=345M : String, which model to use    
    """
    
    enc = encoder.get_encoder(model_name)    
    tf.saved_model.save(enc, 'tmp/tfmodel')
        

if __name__ == '__main__':
    fire.Fire(interact_model)
