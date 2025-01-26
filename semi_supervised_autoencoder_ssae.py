import numpy as np
from keras import models,layers, regularizers, Input, optimizers
import tensorflow as tf
import pandas as pd
from keras.src.ops import squeeze, expand_dims
from tensorflow.python.keras.utils.np_utils import to_categorical


#Reconstruction Loss
def reconstruction_loss(y_true,y_pred):
  loss=tf.keras.losses.MeanSquaredError()(y_true,y_pred)
  return loss
#Classification Loss
def classification_loss(y_true,y_pred):
    loss=tf.keras.losses.BinaryCrossentropy()
    n_el = y_true.shape[0]
    total_loss=0
    skipped = 0
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    for pred, gt in zip(y_pred,y_true):
        if not np.isnan(gt[0]):
          total_loss += loss(pred, gt)
        else:
            skipped += 1
    # If we want to add the mean of the other loss
    # uncomment the line 30, 31, 32
    # mean of loss
    #mean_loss=total_loss/n_el
    #for _ in range(skipped):
    #    total_loss += mean_loss

    return total_loss

def multiclass_classification_loss(y_true,y_pred):
    loss=tf.keras.losses.CategoricalCrossentropy()
    n_class = y_pred.shape[1]
    n_el = y_true.shape[0]
    total_loss=0
    skipped = 0
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    for pred, gt in zip(y_pred,y_true):
        if not np.isnan(gt[0]):
            if len(gt) == 1:
                class_gt = to_categorical(gt[0], n_class)
                total_loss += loss(pred, class_gt)
            else:
                total_loss += loss(pred, gt)
        else:
            skipped += 1
    # mean of loss
    # mean_loss=total_loss/n_el
    # for _ in range(skipped):
    #    total_loss += mean_loss

    return total_loss

def ssae(input_shape,encoder_shape0,encoder_shape1,alpha, n_class=2):
    assert n_class >= 2, 'Number of classes must be greater than 2.'

    input = Input(shape=(input_shape,))

    encoder0=layers.Dense(encoder_shape0, activation='relu')(input)

    encoder1=layers.Dense(encoder_shape1, activation='relu', name='layer_reduced',
                            kernel_regularizer=regularizers.l2(0.00001))(encoder0)


    # decoded9 = layers.Dense(64, activation='linear')(encoded9)

    decoder= layers.Dense(input_shape, activation='linear', name='decoded_output')(encoder1)

    #
    if n_class == 2:
        classification = layers.Dense(1, activation='sigmoid', name='classification_output')(encoder1)
    else:
        classification = layers.Dense(n_class, activation='softmax', name='classification_output')(encoder1)

    output = decoder, classification
    model = models.Model(inputs=input, outputs=output)

    if n_class == 2:
        model.compile(
            optimizer='adam',
           loss={
               'decoded_output': reconstruction_loss,
               'classification_output': classification_loss
           },
           loss_weights={
               'decoded_output': alpha,
               'classification_output': 1 - alpha
           },
           metrics={
               'decoded_output': tf.keras.metrics.MeanSquaredError(),
               'classification_output': tf.keras.metrics.BinaryAccuracy()
           }
        )
    else:
        model.compile(
            optimizer='adam',
            loss={
                'decoded_output': reconstruction_loss,
                'classification_output': multiclass_classification_loss
            },
            loss_weights={
                'decoded_output': alpha,
                'classification_output': 1 - alpha
            },
            metrics={
                'decoded_output': tf.keras.metrics.MeanSquaredError(),
                'classification_output': tf.keras.metrics.CategoricalAccuracy()
            }
        )

    return model

def load_dataset(dataset_name):
    if dataset_name == 'dataset1':
        return pd.read_csv('../data/df_autismo_clr.csv')
    elif dataset_name == 'dataset2':
        return pd.read_csv('../data/df_completo_filtrato_tesi.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")