from __future__ import print_function, division
import os
import datetime
import torch
import pandas as pd
import time
#from skimage import io, transform
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cProfile
import random
#name = name
from scipy import stats
#import hickle
import pickle
import h5py 
#self._inference_network = None
trace_cache_path = None
main_trace_cache_path = None
trace_cache = []
trace_cache_discarded_file_names = []
num_buckets = None
num_files_per_bucket=None
UseBuckets = True
discard_source = False
#batch_size=1
#rootdir='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache'
def use_trace_cache(trace_cache_path):
    main_trace_cache_path = trace_cache_path #never change it
    if UseBuckets:
        dirlist=[]
        dirlist=[x for x in os.listdir(trace_cache_path) if x.startswith('bucket_')]
        num_buckets = len(dirlist)
        num_files=0
        count=0
        for dirname in dirlist:
            temp_trace_cache_path = os.path.join(trace_cache_path, dirname)
            #print (trace_cache_path)
            num_files= len(trace_cache_current_files(temp_trace_cache_path))
            count+= num_files
            global num_files_per_bucket
            num_files_per_bucket=num_files #each bucket has the same number of files

        print('Monitoring bucket folders under trace cache (currently with {} files) in {} buckets at {}'.format(count, num_buckets, trace_cache_path))
        #trace_cache_path = trace_cache_path #base directory for traces
    else:#original code 
        trace_cache_path = trace_cache_path
        num_files = len(trace_cache_current_files(trace_cache_path))
        print('Monitoring trace cache (currently with {} files) at {}'.format(num_files, trace_cache_path))

def trace_cache_current_files(trace_cache_path):
    files = [name for name in os.listdir(trace_cache_path)]
    files = list(map(lambda f: os.path.join(trace_cache_path, f), files))
    for discarded_file_name in trace_cache_discarded_file_names:
        if discarded_file_name in files:
            files.remove(discarded_file_name)
    return files

def Pyprob_IO_Kernel(batch_size, bucket_idx, rootdir,savedir=None,pickle_module=None):
    trace_cache_path=rootdir
    use_trace_cache(trace_cache_path)
    global UseBuckets,num_files_per_bucket,trace_cache
    if UseBuckets:
        worldsize=1 # for serial version
        #dist.get_world_size() # for parallel version
        num_iter_per_bucket = int(num_files_per_bucket/(batch_size * worldsize))
    if bucket_idx is not None:
        trace_cache_path = rootdir + '/bucket_' + str(bucket_idx)
    else: #validation set
        print('randomly pick one bucket for validation set')
        trace_cache_path = rootdir + '/bucket_' + str(random.randrange(10))
        print('trace_cache_path used for validation is {}'.format(trace_cache_path))
    if discard_source:
        trace_cache = []
    loaded=0
    f=h5py.File('trace_'+str(batch_size)+'.h5','a') # create a new file to store the whole batch
    #dset_map_controlled = f.create_dataset('sample_map_controlled', maxshape = (None))
    
    dset_compound = f.create_dataset('trace_dset', shape=(None, ), dtype=dset_cmpd_type)
    while len(trace_cache) < batch_size:
                        #print("current trace_cache_path is {}".format(self._trace_cache_path))
            current_files = trace_cache_current_files(trace_cache_path)
                        #print("current_files number={}".format(len(current_files)))
                        #print("size={}".format(size))
            chosen_files= random.sample(current_files, batch_size)
            if discard_source:
                trace_cache_discarded_file_names.extend(chosen_files)
            time_load_start=time.time()
            #Unique samples
            samples_controlled={}
            samples_uncontrolled={}
            samples_replaced={}
            samples_observed={}
            #log_prob=[]
            #Number of unique samples
            controlled_number=0
            uncontrolled_number=0
            observed_number=0
            replaced_number=0
            #Length of address string length
            controlled_length=list()
            uncontrolled_length=list()
            observed_length=list()
            replaced_length=list()
            #log_importance_weight=[]
            for file in chosen_files:
                loaded+=1
                ####### TRACE LOAD ##############
                ####### TRACE LOAD ##############
                ####### TRACE LOAD ##############
                new_trace= torch.load(file) #the file is already in .pt format and not need decompress
                new_file_name=file.split('/')[-1]+'_'+str(loaded)+'.pt'
                trace_cache.append(new_trace)
                #for x in ['samples','samples_observed','sample_replaced','sample_uncontrolled']:
                controlled_number+=len(new_trace.samples)
                uncontrolled_number+=len(new_trace.samples_uncontrolled)
                observed_number+=len(new_trace.samples_observed)
                replaced_number+=len(new_trace.samples_replaced)
                for i in range(len(new_trace.samples)):
                    isap=new_trace.samples[i]
                    if isap.address not in samples_controlled:
                       samples_controlled[isap.address]=1
                       controlled_length.append(len(isap.address))
                    else:
                       samples_controlled[isap.address]+=1
                for i in range(len(new_trace.samples_uncontrolled)):
                    isap=new_trace.samples_uncontrolled[i]
                    if isap.address not in samples_uncontrolled:
                       samples_uncontrolled[isap.address]=1
                       uncontrolled_length.append(len(isap.address))
                    else:
                       samples_uncontrolled[isap.address]+=1
                for i in range(len(new_trace.samples_observed)):
                    isap=new_trace.samples_observed[i]
                    if isap.address not in samples_observed:
                       samples_observed[isap.address]=1
                       observed_length.append(len(isap.address))
                    else:
                       samples_observed[isap.address]+=1
                for i in range(len(new_trace.samples_replaced)):
                    isap=new_trace.samples_replaced[i]
                    if isap.address not in samples_replaced:
                       samples_replaced[isap.address]=1
                       replaced_length.append(len(isap.address))
                    else:
                       samples_replaced[isap.address]+=1

            #for k,v in samples_controlled.items():
            #    print("{} = {}".format(k, v))
            #for k,v in samples_observed.items():
            #    print("{} = {}".format(k, v))
            #for k,v in samples_replaced.items():
            #    print("{} = {}".format(k, v))

            time_load_end=time.time()
            address_base={}
            log_prob=[]
            log_importance_weight=[]
    traces = trace_cache[0:batch_size]
    #trace_cache[0:batch_size] = []
    print ('loaded :%d'%loaded)
    print ('bytes in memory:%f,%f'%(sys.getsizeof(traces), sys.getsizeof(trace_cache)))
    trace_cache[0:batch_size] = []
    return traces, batch_size,sys.getsizeof(traces)

def benchmark(batch_size,buckidx,rdir,savedir=None,pickle_module=None):
    import time
    start = time.time()
    Pyprob_IO_Kernel(batch_size, buckidx, rdir,savedir,pickle_module)
    end = time.time()
    print ('Number of traces loaded:%d, total time:%f'%(batch_size,end-start))

if __name__ == "__main__":
   if (len(sys.argv)!=3):
        print ("args: batch_size, bucket")
        exit()
   print (sys.argv)
   batch_size = int(sys.argv[1]) #1244
   bucket = int(sys.argv[2]) # must > =0
   if (bucket <0 or batch_size <0 or batch_size > 12440):
        print ('buckets [1-10], batch_size [1,12440]')
        exit()
   rdir='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache'
   benchmark(batch_size,bucket,rdir)
   print ('done with bucket %d'%bucket)
