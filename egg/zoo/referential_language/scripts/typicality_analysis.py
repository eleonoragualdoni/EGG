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
from sklearn import metrics

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_typicalities_info", help="directory typicality scores", default = "VG_typicalities/")
    parser.add_argument("--path_prototypes", help="directory prototypes", default = "VG_prototypes/")
    return parser.parse_args()

opts = get_opts()

files_typ = os.listdir(opts.path_typicalities_info)
files_proto = os.listdir(opts.path_prototypes)

typicalities = []
for i in files_typ:
    typicalities.append([i.split(".")[0], 
                         pickle.load(open(opts.path_typicalities_info + "/" + i, "rb"))])
    
info = []
for i in typicalities:
    info.append([i[0], statistics.mean(i[1]), \
                 statistics.variance(i[1]), 
                 statistics.stdev(i[1])])
  
prototypes = {}    
for i in files_proto:
    name = re.findall("(.*).h5", i)[0]
    proto_file = h5py.File(opts.path_prototypes + i, "r")
    prototypes[name] = proto_file[name][:][0]   


# IN-CLASS ANALYSES
df = pd.DataFrame(info, columns = ["category", "mean typicality", "variance", "stdev"])
df.sort_values(by = "variance", ascending = False)
df.head()


# ACROSS-CLASS ANALYSIS
sim = sklearn.metrics.pairwise.cosine_similarity(list(prototypes.values()))
df_sim = pd.DataFrame(sim, columns = prototypes.keys(), index = prototypes.keys())

similarities = []
for i in list(prototypes.keys()): 
    tmp=[]
    for j in list(prototypes.keys()): 
        tmp.append([i, j, df_sim[i][j]])
    similarities.append(tmp)

all_sims = []
for i in similarities:
    for j in i:
        all_sims.append(j)


all_sims.sort(key = lambda x: x[2])
print(all_sims[:60])








