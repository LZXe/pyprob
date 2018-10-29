from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cProfile
import random
import copy
from pympler import asizeof
#name = name
from pyprob.trace import Sample
from pyprob.trace import Trace
from pyprob.distributions import Uniform
from pyprob.distributions import Categorical
from pyprob.distributions import Poisson
from pyprob import util
import pyarrow.plasma as plasma
import pyarrow
import mpi4py as MPI
client = plasma.connect("/tmp/plasma","",0)
#self._inference_network = None
trace_cache_path = None
main_trace_cache_path = None
trace_cache = []
oid = []
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

def Pyprob_IO_Kernel(batch_size, bucket_idx, rootdir):
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
    while len(trace_cache) < batch_size:
                        #print("current trace_cache_path is {}".format(self._trace_cache_path))
            current_files = trace_cache_current_files(trace_cache_path)
                        #print("current_files number={}".format(len(current_files)))
                        #print("size={}".format(size))
            chosen_files= random.sample(current_files, batch_size)
            if discard_source:
                trace_cache_discarded_file_names.extend(chosen_files)
            global oid
            for file in chosen_files:
                loaded+=1
                new_trace= torch.load(file) #the file is already in .pt format and not need decompress
                #new_file_name=file.split('/')[-1]+'_'+str(loaded)+'.pt'
                #torch.save(new_trace,new_file_name)
                trace_cache.append(new_trace)
                new_trace_pyarrow = pyarrow_obj(new_trace)
                ioid = plasma_tsave(new_trace_pyarrow)
                print ('put object:%s'%ioid)
                oid.append(ioid)
                
                #if (loaded %100==0):
                    #print (new_trace)
                    #print (new_trace.length)
                    #print (new_trace.samples)
                    #print (new_trace.samples_replaced)
    traces = trace_cache[0:batch_size]
    #trace_cache[0:batch_size] = []
    print ('loaded :%d'%loaded)
    print ('bytes in memory:%f,%f'%(sys.getsizeof(traces), sys.getsizeof(trace_cache)))
    trace_cache[0:batch_size] = []
    return traces, batch_size,sys.getsizeof(traces)
def pyarrow_obj(traceobj):
    buf = pyarrow.serialize(traceobj,context=context).to_buffer()
    return buf

context = pyarrow.SerializationContext()
context.register_type(Sample, 'Sample',pickle=True)
context.register_type(Trace, 'Trace',pickle=True)
context.register_type(torch.Tensor, 'torch.Tensor',pickle=True)
context.register_type(Uniform, 'Uniform',pickle=True)
context.register_type(Categorical, 'Categorical',pickle=True)
context.register_type(Poisson,'Poisson',pickle=True)

def plasma_tsave(traceobj):
    object_id = client.put(traceobj)
    return object_id
def list_traces():
    print (client.list())   
def benchmark(batch_size,buckidx,rdir):
    import time
    start = time.time()
    t,b,s=Pyprob_IO_Kernel(batch_size, buckidx, rdir)
    end = time.time()
    print ('Number of traces loaded:%d, time:%f'%(batch_size,end-start))
    print ('Trace 0:')
    print (t[0])
    trace1=copy.deepcopy(t[0])
    print (asizeof.asizeof(trace1))
def plasma_tload(oid):
    data = client.get(oid)
    return data
if __name__ == "__main__":
   if (len(sys.argv)!=3):
        print ("args: batch_size, number_buckets")
        exit()
   print (sys.argv)
   batch_size = int(sys.argv[1]) #1244
   num_bucket = int(sys.argv[2]) # must > 0
   if (num_bucket <1 or batch_size <0 or batch_size > 12440):
        print ('buckets [1-10], batch_size [1,12440]')
        exit()
   rdir='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache'
   #range_bk = range(1,10)
   #print (range_bk)
   print (num_bucket)
   buckets_random =  random.sample(range(1,10), num_bucket)
   for i in buckets_random:
        #buckidx=np.random.randint(1,10,dtype='int')
        buckidx = i
        benchmark(batch_size,buckidx,rdir)
        print ('done with bucket %d'%buckidx)
   #list_traces()
   data = plasma_tload(oid[0])
   print ('get obj:%s'%oid[0])
   print (data)
   buf = context.deserialize(data)
   print (buf)
   buf1=copy.deepcopy(buf)
   print (asizeof.asizeof(buf1))
