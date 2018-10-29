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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
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
        return num_files_per_bucket
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

def Pyprob_IO_Kernel(batch_size, bucket_idx, rootdir,rank=None):
    trace_cache_path=rootdir
    num_files_per_bucket = use_trace_cache(trace_cache_path)
    trace_cache=[]
    UseBuckets=True
    if UseBuckets:
        worldsize=1 # for serial version
        #dist.get_world_size() # for parallel version
        num_iter_per_bucket = int(num_files_per_bucket/(batch_size * worldsize))
    if bucket_idx is not None:
        #if rank is not None: 
        #   bucket_idx = int (rank % 10 +1)
        if rank == 0: 
           bucket_idx=3
        if rank == 1: 
           bucket_idx=4
        trace_cache_path = rootdir + '/bucket_' + str(bucket_idx)
    else: #validation set
        print('randomly pick one bucket for validation set')
        trace_cache_path = rootdir + '/bucket_' + str(random.randrange(10))
        print('trace_cache_path used for validation is {}'.format(trace_cache_path))
    print ('rank:%d,path:%s'%(rank,trace_cache_path))
    if discard_source:
        trace_cache = []
    loaded=0
    while len(trace_cache) < batch_size:
                        #print("current trace_cache_path is {}".format(self._trace_cache_path))
            current_files = trace_cache_current_files(trace_cache_path)
                        #print("current_files number={}".format(len(current_files)))
                        #print("size={}".format(size))
            print ('length of current files:%d,local batch:%d'%(len(current_files),batch_size))
            chosen_files= random.sample(current_files, batch_size)
            if discard_source:
                trace_cache_discarded_file_names.extend(chosen_files)
            #global oid
            for file in chosen_files:
                loaded+=1
                print ('file:%s'%file.split('/')[-1])
                new_trace= torch.load(file) #the file is already in .pt format and not need decompress
                #new_file_name=file.split('/')[-1]+'_'+str(loaded)+'.pt'
                #torch.save(new_trace,new_file_name)
                trace_cache.append(new_trace)
                new_trace_pyarrow = pyarrow_obj(new_trace)
                ioid=0
                try:
                   ioid = plasma_tsave(new_trace_pyarrow)
                except Exception as e: 
                   print ('error in plasma save:%s,file:%s'%(e,file.split('/')[-1]))
                   list_traces()
                   continue
                if ioid ==0:
                   list_traces()
                   continue
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
    return traces, batch_size,sys.getsizeof(traces),oid
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
def benchmark(batch_size,buckidx,rdir,rank=None,rank_check=None):
    import time
    start = time.time()
    buckidxx=rank % 10 # temorary solution to avoid collision
    print ('rank:%d,bucket:%d'%(rank,buckidxx)) 
    t,b,s,oids=Pyprob_IO_Kernel(batch_size, buckidxx, rdir,rank)
    end = time.time()
    print ('Number of traces loaded:%d, time:%f'%(batch_size,end-start))
    if rank==rank_check and rank is not None and rank_check is not None:
       print ('Trace 0:')
       print (t[0])
       trace1=copy.deepcopy(t[0])
       print (asizeof.asizeof(trace1))
    return oids
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
   print ('number of buckets:%d'%num_bucket)
   print ('number of ranks:%d'%(size))
   local_batch = int(batch_size/size) # e.g., global batch is 1024, mpi rank is 32, then local batch is 32
   print ('local batch:%d'%local_batch)
   buckets_random =  random.sample(range(1,10), num_bucket)
   x=0
   y=size/2
   for i in buckets_random:
        #buckidx=np.random.randint(1,10,dtype='int')
        buckidx = i
        oids=benchmark(local_batch,buckidx,rdir,rank, y)
        print ('done with bucket %d'%buckidx)
   #list_traces()
   # load one object from rank x to rank y
   print ('loading one object from rank %d to rank %d'%(y,x))
   #data = plasma_tload(oid[0],x,y)
   if rank==y:
      comm.send(oid,dest=x)
   elif rank==x:
      oid_y = comm.recv(source=y)   
      data = plasma_tload(oid_y[0])
      print ('get obj:%s'%oid_y[0])
      print (data)
      buf = context.deserialize(data)
      print (buf)
      buf1=copy.deepcopy(buf)
      print (asizeof.asizeof(buf1))
