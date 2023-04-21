#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system(' pip install mne')


# In[3]:


from glob import glob
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


glob('Downloads/omaima10/*.edf')


# In[5]:


all_file=glob('Downloads/omaima10/*.edf')
print(len(all_file))


# In[64]:


healthy_file=[i for i in all_file if 'h' in i.split('\\')[1]]
patient_file=[i for i in all_file if 's' in i.split('\\')[1]]
print(len(healthy_file),len(patient_file))


# In[71]:


def read_data(file_path):
    data=mne.io.read_raw_edf(file_path,preload=True)
    data.set_eeg_reference()
    data.filter(l_freq=0.5,h_freq=45)
    epochs=mne.make_fixed_length_epochs(data,duration=25,overlap=0)
    array=epochs.get_data()
    return array
    


# In[72]:


sample_data=read_data(healthy_file[0])


# In[73]:


sample_data.shape


# In[74]:


get_ipython().run_cell_magic('capture', '', 'control_epochs_array=[read_data(i) for i in healthy_file]\npatient_epchs_array=[read_data(i) for i in healthy_file]')


# In[75]:


control_epochs_array[0].shape,control_epochs_array[1].shape


# In[76]:


control_epoch_labels= [len(i)*[0] for i in control_epochs_array]
patient_epoch_labels=[len(i)*[1] for i in patient_epchs_array]
len(control_epoch_labels),len(patient_epoch_labels)


# In[77]:


data_list=control_epochs_array+patient_epchs_array
label_list=control_epoch_labels+patient_epoch_labels


# In[78]:


group_list=[[i]* len(j) for i,j in enumerate (data_list)]
len(group_list)


# In[79]:


data_array=np.vstack(data_list)
label_array=np.hstack(label_list)
group_array=np.hstack(group_list)
print(data_array.shape,label_array.shape,group_array.shape)


# In[80]:


from scipy import stats
def mean(x):
    return np.mean(x,axis=-1)
def std(x):
    return np.std(x,axis=-1)
def ptp(x):
    return np.ptp(x,axis=-1)
def var (x):
    return np.var(x,axis=-1)
def minim(x):
    return np.min(x,axis=-1)
def maxim(x):
    return np.max(x,axis=-1)
def argminim(x):
    return np.argmin(x,axis=-1)
def argmaxim(x):
    return np.argmax(x,axis=-1)
def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))
def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)
def skewness(x):
    return stats.skew(x,axis=-1)
def kurtosis(x):
    return stats.kurtosis(x,axis=-1)
def concatenate_features(x):
    return np.concatenate((mean(x),std(x),ptp(x),var(x),minim(x),maxim(x),argminim(x),argmaxim(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x)),axis=-1)


# In[81]:


features=[]
for d in data_array:
    features.append(concatenate_features(d))


# In[82]:


features_array=np.array(features)
features_array.shape


# In[83]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,GridSearchCV


# In[84]:


clf=LogisticRegression()
gkf=GroupKFold(5)
pipe=Pipeline([('scaler',StandardScaler()),('clf',clf)])
pram_grid={'clf__C':[0.1,0.5,0.7,1,3,5,7]}
gscv=GridSearchCV(pipe,pram_grid,cv=gkf,n_jobs=12)
gscv.fit(features_array,label_array,groups=group_array)


# In[85]:


gscv.best_score_


# In[61]:





# In[ ]:




