import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import model_dict, Distiller
from datasets import cifar

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cifar100", 
                        choices=["cifar100", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--tmodel", "--teacher", type=str, default="resnet50", 
                        choices=["resnet50"])
    parser.add_argument("--smodel", "--student", type=str, default="mobilenetv2",
                        choices=["mobilenetv2"])
    parser.add_argument("--tpath", "--tweights", type="str")
    parser.add_argument("-T", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--savepath", type=str, default="save")
    
    opt = parser.parse_args()
    
    opt.save_folder = os.path.join(
        opt.savepath, f"{opt.smodel}_{opt.tmodel}_{opt.epochs}_{opt.seed}"
    )
    
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    opt = parse_option()
    set_seed(opt.seed)
    
    # prepare for model
    teacher_model = model_dict[opt.tmodel]
    student_model = model_dict[opt.smodel]
    teacher_model.load_weights(opt.tpath)
    distiller = Distiller(teacher=teacher_model, 
                          student=student_model)
    
    # prepare for data
    if opt.dataset == "cifar100" or opt.dataset == "cifar10":
        train_ds, test_ds = cifar(opt.dataset, opt.batch_size)
    else:
        raise NotImplementedError
    
    optimizer = keras.optimizers.Adam()
    ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    kl_loss = keras.losses.KLDivergence()
    
    distiller.compile(
        optimizer = optimizer,
        metrics = ["accuracy"], 
        ce_loss=ce_loss,
        kd_loss=kl_loss,
        temperature=opt.T,
        alpha=opt.alpha
    )
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_best = keras.callbacks.ModelCheckpoint(
        os.path.join(opt.save_folder, f"{now}.h5"),
        save_best_only = True
    )

    distiller.fit(
        train_ds, 
        epochs = opt.epochs,
        validation_data = test_ds,
        callbacks= [save_best]
    )
    


if __name__ == "__main__":
    main()
