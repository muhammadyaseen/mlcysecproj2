#!/usr/bin/env python
# coding: utf-8

# # ML in Cybersecurity: Project II
# 
# ## Team
#   * **Team name**:  MMM
#   * **Members**:  Maria Sargsyan (<email here>), Muneeb Aadil (2581794, maadil@mpi-inf.mpg.de), Muhammad Yaseen (2577833, myaseen@mpi-inf.mpg.de).
# 
# 
# ## Logistics
#   * **Due date**: 28th November 2019, 13:59:59 (right before the lecture)
#   * Email the completed notebook to: `mlcysec_ws1920_staff@lists.cispa.saarland`
#   * Complete this in **teams of 3**
#   * Feel free to use the course [mailing list](https://lists.cispa.saarland/listinfo/mlcysec_ws1920_stud) to discuss.
#   
# ## Timeline
#   * 14-Nov-2019: Project 2 hand-out
#   * **28-Nov-2019** (13:59:59): Email completed notebook
#   * 5-Nov-2019: Project 2 discussion and summary
#   
#   
# ## About this Project
# In this project, you will explore an application of ML to a popular task in cybersecurity: malware classification.
# You will be presented with precomputed behaviour analysis reports of thousands of program binaries, many of which are malwares.
# Your goal will be train a malware detector using this behavioural reports.
# 
# 
# ## A Note on Grading
# The grading for this project will depend on:
#  1. Vectorizing Inputs
#    * Obtaining a reasonable vectorized representations of the input data (a file containing a sequence of system calls)
#    * Understanding the influence these representations have on your model
#  1. Classification Model  
#    * Following a clear ML pipeline
#    * Obtaining reasonable performances (>60\%) on held-out test set
#    * Choice of evaluation metric
#    * Visualizing loss/accuracy curves
#  1. Analysis
#    * Which methods (input representations/ML models) work better than the rest and why?
#    * Which hyper-parameters and design-choices were important in each of your methods?
#    * Quantifying influence of these hyper-parameters on loss and/or validation accuracies
#    * Trade-offs between methods, hyper-parameters, design-choices
#    * Anything else you find interesting (this part is open-ended)
# 
# 
# ## Grading Details
#  * 40 points: Vectorizing input data (each input = behaviour analysis file in our case)
#  * 40 points: Training a classification model
#  * 15 points: Analysis/Discussion
#  * 5 points: Clean code
#  
# ## Filling-in the Notebook
# You'll be submitting this very notebook that is filled-in with your code and analysis. Make sure you submit one that has been previously executed in-order. (So that results/graphs are already visible upon opening it). 
# 
# The notebook you submit **should compile** (or should be self-contained and sufficiently commented). Check tutorial 1 on how to set up the Python3 environment.
# 
# 
# **The notebook is your project report. So, to make the report readable, omit code for techniques/models/things that did not work. You can use final summary to provide report about these codes.**
# 
# It is extremely important that you **do not** re-order the existing sections. Apart from that, the code blocks that you need to fill-in are given by:
# ```
# #
# #
# # ------- Your Code -------
# #
# #
# ```
# Feel free to break this into multiple-cells. It's even better if you interleave explanations and code-blocks so that the entire notebook forms a readable "story".
# 
# 
# ## Code of Honor
# We encourage discussing ideas and concepts with other students to help you learn and better understand the course content. However, the work you submit and present **must be original** and demonstrate your effort in solving the presented problems. **We will not tolerate** blatantly using existing solutions (such as from the internet), improper collaboration (e.g., sharing code or experimental data between groups) and plagiarism. If the honor code is not met, no points will be awarded.
# 
#  
#  ## Versions
#   * v1.1: Updated deadline
#   * v1.0: Initial notebook
#   
#   ---

# In[1]:


import time 

import numpy as np 
import matplotlib.pyplot as plt 

import json 
import time 
import pickle 
import sys 
import csv 
import os 
import os.path as osp 
import shutil 
import pathlib
from pathlib import Path

from IPython.display import display, HTML
 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots 
plt.rcParams['image.interpolation'] = 'nearest' 
plt.rcParams['image.cmap'] = 'gray' 
 
# for auto-reloading external modules 
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Some suggestions of our libraries that might be helpful for this project
from collections import Counter          # an even easier way to count
from multiprocessing import Pool         # for multiprocessing
from tqdm import tqdm_notebook as tqdm                    # fancy progress bars

# Load other libraries here.
# Keep it minimal! We should be easily able to reproduce your code.
# We only support sklearn and pytorch.

# We preload pytorch as an example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import pandas as pd


# In[3]:


compute_mode = 'cpu'

if compute_mode == 'cpu':
    device = torch.device('cpu')
elif compute_mode == 'gpu':
    # If you are using pytorch on the GPU cluster, you have to manually specify which GPU device to use
    # It is extremely important that you *do not* spawn multi-GPU jobs.
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'    # Set device ID here
    device = torch.device('cuda')
else:
    raise ValueError('Unrecognized compute mode')


# # Setup
# 
#   * Download the datasets: [train](https://nextcloud.mpi-klsb.mpg.de/index.php/s/pJrRGzm2So2PMZm) (128M) and [test](https://nextcloud.mpi-klsb.mpg.de/index.php/s/zN3yeWzQB3i5WqE) (92M)
#   * Unpack them under `./data/train` and `./data/test`

# In[4]:


# Check that you are prepared with the data
get_ipython().system(" printf '# train examples (Should be 13682) : '; ls data/train | wc -l")
get_ipython().system(" printf '# test  examples (Should be 10000) : '; ls data/test | wc -l")


# Now that you're set, let's briefly look at the data you have been handed.
# Each file encodes the behavior report of a program (potentially a malware), using an encoding scheme called "The Malware Instruction Set" (MIST for short).
# At this point, we highly recommend you briefly read-up Sec. 2 of the [MIST](http://www.mlsec.org/malheur/docs/mist-tr.pdf) documentation.
# 
# You will find each file named as `filename.<malwarename>`:
# ```
# » ls data/train | head
# 00005ecc06ae3e489042e979717bb1455f17ac9d.NothingFound
# 0008e3d188483aeae0de62d8d3a1479bd63ed8c9.Basun
# 000d2eea77ee037b7ef99586eb2f1433991baca9.Patched
# 000d996fa8f3c83c1c5568687bb3883a543ec874.Basun
# 0010f78d3ffee61101068a0722e09a98959a5f2c.Basun
# 0013cd0a8febd88bfc4333e20486bd1a9816fcbf.Basun
# 0014aca72eb88a7f20fce5a4e000c1f7fff4958a.Texel
# 001ffc75f24a0ae63a7033a01b8152ba371f6154.Texel
# 0022d6ba67d556b931e3ab26abcd7490393703c4.Basun
# 0028c307a125cf0fdc97d7a1ffce118c6e560a70.Swizzor
# ...
# ```
# and within each file, you will see a sequence of individual systems calls monitored duing the run-time of the binary - a malware named 'Basun' in the case:
# ```
# » head data/train/000d996fa8f3c83c1c5568687bb3883a543ec874.Basun
# # process 000006c8 0000066a 022c82f4 00000000 thread 0001 #
# 02 01 | 000006c8 0000066a 00015000
# 02 02 | 00006b2c 047c8042 000b9000
# 02 02 | 00006b2c 047c8042 00108000
# 02 02 | 00006b2c 047c8042 00153000
# 02 02 | 00006b2c 047c8042 00091000
# 02 02 | 00006b2c 047c8042 00049000
# 02 02 | 00006b2c 047c8042 000aa000
# 02 02 | 00006b2c 047c8042 00092000
# 02 02 | 00006b2c 047c8042 00011000
# ...
# ```
# (**Note**: Please ignore the first line that begins with `# process ...`.)
# 
# Your task in this project is to train a malware detector, which given the sequence of system calls (in the MIST-formatted file like above), predicts one of 10 classes: `{ Agent, Allaple, AutoIt, Basun, NothingFound, Patched, Swizzor, Texel, VB, Virut }`, where `NothingFound` roughly represents no malware is present.
# In terms of machine learning terminology, your malware detector $F: X \rightarrow Y$ should learn a mapping from the MIST-encoded behaviour report (the input $x \in X$) to the malware class $y \in Y$.
# 
# Consequently, you will primarily tackle two challenges in this project:
#   1. "Vectorizing" the input data i.e., representing each input (file) as a tensor
#   1. Training an ML model
#   
# 
# ### Some tips:
#   * Begin with an extremely simple representation/ML model and get above chance-level classification performance
#   * Choose your evaluation metric wisely
#   * Save intermediate computations (e.g., a token to index mapping). This will avoid you parsing the entire dataset for every experiment
#   * Try using `multiprocessing.Pool` to parallelize your `for` loops

# ---

# # 1. Vectorize Data

# ## 1.a. Load Raw Data

# In[5]:


CLASSES = ['Agent', 'Allaple', 'AutoIt', 'Basun', 'NothingFound', 'Patched', 'Swizzor', 'Texel', 'VB', 'Virut']

# We wrote our own PyTorch data loader


# ## 1.b. Vectorize: Setup
# 
# Make one pass over the inputs to identify relevant features/tokens.
# 
# Suggestion:
#   - identify tokens (e.g., unigrams, bigrams)
#   - create a token -> index (int) mapping. Note that you might have a >10K unique tokens. So, you will have to choose a suitable "vocabulary" size.

# ## 1.c. Vectorize Data
# 
# Use the (token $\rightarrow$ index) mapping you created before to vectorize your data

# In[11]:


#
#
# ------- Your Code -------
#
#


# In[14]:


# (a) You can use torch.utils.data.TensorDataset to represent the tensors you created previously

# (b) Store your datasets to disk so that you do not need to precompute it every time


# # 2. Train Model
# 
# You will now train an ML model on the vectorized datasets you created previously.
# 
# _Note_: Although I often refer to each input as a 'vector' for simplicity, each of your inputs can also be higher dimensional tensors.

# ## 2.a. Helpers

# In[15]:


# WARNING: THIS CLASSES LIST IS COPIED FROM MLW_LOADER.
CLASSES = ["NothingFound", "Basun", "Agent", "Allaple", "AutoIt", 
           "Patched", "Swizzor", "Texel", "VB", "Virut"]

def evaluate_preds(y_gt, y_pred):
    pass

def save_model(model, out_path):
    pass


def find_class_occurences(dir_path='./data/train'):
    """
    Returns a dictionary with having class name as key, and the number of instances
    having the class as its corresponding key.
    
    Args:
        dir_path (string): folder containing programs' stack trace.
    
    Returns:
        out (dict): (k,v) pairs where k = class name, v = number of occurences of that class.
    """
    out = dict()
    for program_file in tqdm(os.listdir(dir_path)):
        class_name = program_file.split('.')[-1]
        out[class_name] = out.get(class_name, 0) + 1
    return out
        
def get_class_weights(occ):
    """
    Given class occurences dictionary, it returns a numpy array containing the respective
    weight for each class.
    
    Args:
        occ (dict): (k,v) pairs where k = class name, v = number of occurences of that class.
        
    Returns:
        out (np.array) = an array of shape (num_classes,) containing the weight of each class.
    """
    # finding total number of examples in the set.
    total = 0.0
    for _, v in occ.items():
        total += v
    
    out = np.zeros((len(CLASSES), ))
    for k, v in occ.items():
        weight = 1. / (occ[k] / total)
        out[CLASSES.index(k)] = weight
    
    return out


# In[16]:


class_occ = find_class_occurences(dir_path='./data/train')
class_weights = get_class_weights(class_occ)


# ## 2.a.1 Define Dataset and Transformation to use.

# In[17]:


# load dataset.
import malware_loader as mlw_loader
import malware_transforms as mlw_transforms

vectorizer = mlw_transforms.CategoryCount()


train_dataset = mlw_loader.MalwareDataset(root_dir='./data', dataset_type='train', transform=vectorizer)
test_dataset = mlw_loader.MalwareDataset(root_dir='./data', dataset_type='test', transform=vectorizer)
val_dataset = mlw_loader.MalwareDataset(root_dir='./data', dataset_type='val', transform=vectorizer)


# In[18]:


# setting some dataset dependent features.
num_classes = train_dataset.num_classes
num_feats = vectorizer.num_feats


# ## 2.b. Define Model

# Describe your model here.

# In[19]:


# please see models.py file for models definition

import anns 


# ## 2.c. Set Hyperparameters

# In[20]:


# Define your hyperparameters here

# model instantiation

expt_name = 'baseline.model'

hidden_layers = [128, 256, 128, 64, 32, 16]

# Optimization
n_epochs = 25
batch_size = 32
lr = 5e-3
momentum = 0.9
num_workers = 4


# ## 2.d. Train your Model

# In[ ]:





# In[21]:


# Feel free to edit anything in this block
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = anns.ANN(input_dim=num_feats, hidden_layers=hidden_layers,
                 output_dim=num_classes).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# loss function
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))


# In[31]:


def train_epoch(loader, model, criterion, optimizer, device, print_every=10):
    """Trains the model for one epoch"""
    model.train()
    loader_iterable = tqdm(loader)

    running_loss = 0.0

    for i, sample in enumerate(loader_iterable):
        X_batch, Y_batch = sample['trace'], sample['label_idx']
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred = model(X_batch)

        loss = criterion(Y_pred, Y_batch)
        running_loss = running_loss + loss.data
        
        if (i % print_every) == 0:
            print("Loss at iteration {}: {}".format(i, loss.data))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = running_loss / len(loader_iterable)
    return mean_loss


# In[32]:


def validate_epoch(loader, model, criterion, device, print_every=10):
    """Validates the trained model"""
    model.eval()
    loader_iterable = tqdm(loader)
    running_loss = 0.0

    for i, sample in enumerate(loader_iterable):
        X_batch, Y_batch = sample['trace'], sample['label_idx']
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_pred = model(X_batch)

        loss = criterion(Y_pred, Y_batch)
        running_loss += loss.data

        if (i % print_every) == 0:
            print("Loss at iteration {}: {}".format(i, loss.data))

    mean_loss = running_loss / len(loader_iterable)
    return mean_loss


# In[30]:


train_losses, test_losses = [], []
for epoch in range(n_epochs):
    print("Epoch [{} / {}]".format(epoch+1, n_epochs))
    print("Training...")
    mean_loss_train = train_epoch(train_loader, model, criterion, optimizer, device)
    print("Validating...")
    mean_loss_test = validate_epoch(test_loader, model, criterion, device)
    
    train_losses.append(mean_loss_train)
    test_losses.append(mean_loss_test)


# ## 2.e. Evaluate model

# In[48]:


def predict(loader, model):
    model.eval()
    y_preds, y_gts = [], []

    i = 0
    for sample in tqdm(loader):
        X_batch, Y_batch = sample['trace'], sample['label_idx']
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_scores = model(X_batch)
        
        _, Y_pred = Y_scores.max(1)
        
        y_preds.append(Y_pred)
        y_gts.append(Y_batch)
        
        i += 1
        
        if i == 3:
            break
        
    return torch.cat(y_preds), torch.cat(y_gts)

pred_class, actual_class = predict(test_loader, model)


# In[53]:


# Evaluation Metric: F1 score.
f1score = f1_score(actual_class, pred_class.numpy(), average=None)

f1score.mean()


# ## 2.f. Save Model + Data

# In[ ]:


# Feel free to edit anything in this block

save_path = ''
with open(save_path) as fn:
    pickle.dump(model, fn)
    

# model_out_path = '{}.checkpoint.pth'.format(expt_name)
# save_model(model, model_out_path)

eval_data = {
    'epoch': [],
    'train_loss': [],
    'test_loss': [],
    'train_acc': [],
    'test_acc': [],
    ...
}
eval_out_path = '{}.eval.pickle'.format(expt_name)
save_data(eval_data, eval_out_path)


# ---

# # 3. Analysis

# ## 3.a. Summary: Main Results
# 
# If you tried other approaches, summarize their results here.

# |        | Input Representation | Model | Optimizer | Validation Metric | Test Metric |
# |--------|----------------------|-------|-----------|-------------------|-------------|
# | Model1 | Unigram tokens       | MLP   | SGD       | 12.34 %           | 23.45%      |
# | Model2 (this notebook) |                      |       |           |                   |             |
# | ...    |                      |       |           |                   |             |

# ## 3.b. Discussion
# 
# Enter your final summary here.
# 
# For instance, you can address:
# - What was the performance you obtained with the simplest approach?
# - Which vectorized input representations helped more than the others?
# - Which malwares are difficult to detect and why?
# - Which approach do you recommend to perform malware classification?

# In[ ]:




