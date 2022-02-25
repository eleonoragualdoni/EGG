#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:40:05 2021

@author: eleonora
"""

import numpy as np
import os
import pickle
import statistics
import argparse
import pandas as pd
import sklearn
import re
import h5py
import json
from sklearn import metrics
from matplotlib import pyplot as plt

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_objects", help="objects file", default = 'VG_data/objects.json')
    parser.add_argument("--path_image_data", help="image data file", default = 'VG_data/image_data.json')
    parser.add_argument("--path_classes", help="classes file", default = "VG_data/objects_vocab.txt")
    parser.add_argument("--path_typicalities_info", help="directory typicality scores", default = "VG_typicalities/")
    parser.add_argument("--path_prototypes", help="directory prototypes", default = "VG_prototypes/")
    return parser.parse_args()

opts = get_opts()

# load files
files_typ = os.listdir(opts.path_typicalities_info)
files_proto = os.listdir(opts.path_prototypes)
with open(opts.path_objects) as f:
    object_list = json.load(f)
with open(opts.path_image_data) as f:
    image_data = json.load(f) 
obj_classes = open(opts.path_classes).read().split("\n")
  

### CHECK OBJ CLASS FREQUENCY IN THE DATASET  

# clean classes
dic_writings = {}
for i in obj_classes:
    if "," in i:
        splitted = i.split(",")
        dic_writings[splitted[0]] = splitted[0]
        dic_writings[splitted[1]] = splitted[0]

# keep just one writing version
cleaned_classes = []
for i in obj_classes:
    if "," not in i:
        cleaned_classes.append(i)
    else:  
        cleaned_classes.append(dic_writings[i.split(",")[0]]) 

dic_freq_class = {}
for name in cleaned_classes:
    dic_freq_class[name] = 0
    
for im_data, ob_data in list(zip(image_data, object_list)):
    for obj in ob_data['objects']:
        for name in obj['names']:
            if name in cleaned_classes:
                dic_freq_class[name] += 1
            elif name in dic_writings.keys() and dic_writings[name] in cleaned_classes:
                    dic_freq_class[dic_writings[name]] += 1
            else:
                pass

plt.hist(dic_freq_class.values(), bins =500, range = (0,10000))

### TYPICALITY ANALYSIS

typicalities = []
for i in files_typ:
    typicalities.append([i.split(".")[0], pickle.load(open(opts.path_typicalities_info + "/" + i, "rb"))])
    
info = []
for i in typicalities:
    info.append([i[0], statistics.mean(i[1]), \
                 statistics.variance(i[1]), \
                 statistics.stdev(i[1])])
  
prototypes = {}    
for i in files_proto:
    name = re.findall("(.*).h5", i)[0]
    proto_file = h5py.File(opts.path_prototypes + i, "r")
    prototypes[name] = proto_file[name][:][0]   


# IN-CLASS ANALYSES
#info = pickle.load(open("info_typ_VG.pkl", "rb"))
info_freq = []
for i in info:
    if dic_freq_class[i[0]] > 200:
        info_freq.append(i)
df = pd.DataFrame(info_freq, columns = ["category", "mean typicality", "variance", "stdev"])
df.sort_values(by = "variance", ascending = False)



# ACROSS-CLASS ANALYSIS
#prototypes = pickle.load(open("prototypes_VG.pkl", "rb"))
proto_freq = {}
for k,v in prototypes.items():
    if dic_freq_class[k] > 200:
        proto_freq[k] = v
sim = sklearn.metrics.pairwise.cosine_similarity(list(proto_freq.values()))
df_sim = pd.DataFrame(sim, columns = proto_freq.keys(), index = proto_freq.keys())

similarities = []
for i in list(proto_freq.keys()): 
    tmp=[]
    for j in list(proto_freq.keys()): 
        tmp.append([i, j, df_sim[i][j]])
    similarities.append(tmp)

all_sims = []
for i in similarities:
    for j in i:
        all_sims.append(j)


all_sims.sort(key = lambda x: x[2], reverse = True)
all_sims[1005:1025]






