import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from torchvision import transforms, utils

from datasets import *
from nets import *
from functions import *
from trainer import *
import streamlit as st 

from PIL import Image, ImageOps
import numpy as np
st.title("Face Age Progression")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='path to the config file.')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--img_path', type=str, default='./test/input/', help='test image path')
parser.add_argument('--out_path', type=str, default='./test/output/', help='test output path')
parser.add_argument('--target_age', type=int, default=55, help='Age transform target, interger value between 20 and 70')
opts = parser.parse_known_args()[0]

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'))
img_size = (config['input_w'], config['input_h'])

# Initialize trainer
trainer = Trainer(config)
device = torch.device('cpu')
trainer.to(device)

# Load pretrained model 
if opts.checkpoint:
    trainer.load_checkpoint(opts.checkpoint , device)
else:
    trainer.load_checkpoint(log_dir + 'checkpoint', device)

def preprocess(img_name):
    resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1,1,1])
    img_pil = Image.open('input.jpg')
    img_np = np.array(img_pil)
    img = resize(img_pil)
    if img.size(0) == 1:
        img = torch.cat((img, img, img), dim = 0)
    img = normalize(img)
    return img

st.set_option('deprecation.showfileUploaderEncoding', False)
age_int = st.number_input("Please enter your age between 20 to 70 and press enter ")
# Set target age
target_age = int(age_int)
uploaded_file = st.file_uploader("Choose an image...", type="jpg" , use_column_width=True)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('input.jpg')
# Load test image
    img_list = os.listdir(opts.img_path)
    img_list.sort()

    with torch.no_grad():

    #for img_name in img_list:
        #if not img_name.endswith(('png', 'jpg', 'PNG', 'JPG')):
            #print('File ignored: ' + img_name)
            #continue
      image_A = preprocess('input.jpg')
      img_name = str('input.jpg')
      image_A = image_A.unsqueeze(0).to(device)
      age_modif = torch.tensor(target_age).unsqueeze(0).to(device)
      image_A_modif = trainer.test_eval(image_A, age_modif, target_age=target_age, hist_trans=True)  
      utils.save_image(clip_img(image_A_modif), opts.out_path + img_name.split('.')[0] + '_age_' + str(target_age) + '.jpg')

        # Plot manipulated image
      img_out = np.array(Image.open(opts.out_path + img_name.split('.')[0] + '_age_' + str(target_age) + '.jpg'))
      output = Image.open(opts.out_path + img_name.split('.')[0] + '_age_' + str(target_age) + '.jpg')
      plt.axis('off')
      plt.imshow(img_out)
      st.image(output)
      plt.show() 


