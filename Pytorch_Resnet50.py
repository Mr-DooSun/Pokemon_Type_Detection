from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from fastai.vision import *
import fastai

from fastprogress.fastprogress import force_console_behavior

master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

path2zip = 'D:/Python_Project/Pokemon_Type/train_data/'
train_classes = os.listdir('train_data')
batch_size = 32
stage1_epochs = 7
stage2_epochs = 3

def get_parent_dir(f):
    return os.path.split(os.path.split(f)[0])[-1]
            
def train():
    print(train_classes)
    print('number of train_classes', len(train_classes))
        
    np.random.seed(42)
    
    il = (ImageList.from_folder('train_data')
          .filter_by_func(lambda f: get_parent_dir(f) in train_classes)
          .split_by_rand_pct(valid_pct=0.1, seed=42)
          .label_from_func(lambda f: get_parent_dir(f))
          .transform(tfms=get_transforms())
          )
    data = (ImageDataBunch.create_from_ll(il, bs=batch_size,
                                         size=448,
                                         num_workers=4)
                      .normalize(imagenet_stats)
            )
    print(len(data.classes), data.c, len(data.train_ds), len(data.valid_ds))

    
    learn = (cnn_learner(data, models.resnet50, metrics=accuracy)
             .mixup(alpha=0.2)
             )
    learn.loss_func = LabelSmoothingCrossEntropy()
    learn.fit_one_cycle(stage1_epochs, max_lr=1e-3)
    #learn.save('stage-1')
    learn.unfreeze()
    learn.fit_one_cycle(stage2_epochs, max_lr=slice(1e-6,3e-6))
    #learn.save('stage-2')
    # learn.export('210206_model.pkl')
    learn.export(os.path.abspath('/data/dinhson/car/210219_model_resnet50_100epochs.pkl'))
    interp = ClassificationInterpretation.from_learner(learn)
    df_cm = pd.DataFrame(interp.confusion_matrix(), index=data.classes,
                         columns=data.classes)
    df_cm.to_csv('conf_matrix_220302_resnet50_10epochs.csv')
if __name__ == '__main__':
    train()
    
    
    