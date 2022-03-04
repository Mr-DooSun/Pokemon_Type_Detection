# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from fastai.vision import load_learner, open_image

TOP_N_OUTPUT=3

class car_model(object):
    def __init__(self):
        self.model=self.get_classifier("220303_model_resnet50_100epochs.pkl")
        
    def softmax(self,x):
        e_x=np.exp(x-np.max(x))
        return e_x/e_x.sum(axis=0)
    
    def read_image(self, impath):
        return open_image(impath)
    
    def get_classifier(self, weights=None):
        model=load_learner(*os.path.split(weights))
        return model
    
    def predict_images(self, model, img):
        _, _, outputs=model.predict(img)
        classes=model.data.classes
        tmp={}
        softmaxed=self.softmax(outputs.numpy())
        
        for o in softmaxed.argsort()[-3:][::-1]:
            tmp.update({classes[o]:softmaxed[o].item()})
        
        pred=max(tmp, key=lambda k: tmp[k])
        return tmp, '\n Predict: {} and Probability {}'.format(pred, tmp[pred])
    
    def car_model_test(self, imageName):
        im=self.read_image(imageName)
        
        preds=self.predict_images(self.model, im)
        
        return preds

c = car_model()

print(c.car_model_test('test/뚱이.jpg'))