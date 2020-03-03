#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M'    
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use    
    """
    
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    tf.saved_model.save(enc, 'tmp/tfmodel')
        

if __name__ == '__main__':
    fire.Fire(interact_model)
