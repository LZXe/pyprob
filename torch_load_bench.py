from __future__ import print_function, division
import random
import os
import pyprob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import inspect
import difflib
import inspect
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
import time
import torch._utils
import cProfile
from contextlib import closing, contextmanager
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct('=l').size
INT_SIZE = struct.Struct('=i').size
SHORT_SIZE = struct.Struct('=h').size

MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ','
serialized_container_types={}

_package_registry = []
map_location=None

def _check_seekable(f):

    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (str(e) + ". You can only torch.load from a file that is seekable." +
                                " Please pre-load the data into a buffer like io.BytesIO and" +
                                " try to load from it instead.")
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)

def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False

def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)

def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace('Storage', 'Tensor'))
def _is_compressed_file(f):
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False
def default_restore_location(storage, location):
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError("don't know how to restore data location of " +
                        torch.typename(storage) + " (tagged with " +
                        location + ")")

def register_package(priority, tagger, deserializer):
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()
    
def _cpu_tag(obj):
    if type(obj).__module__ == 'torch':
        return 'cpu'


def _cuda_tag(obj):
    if type(obj).__module__ == 'torch.cuda':
        return 'cuda:' + str(obj.get_device())


def _cpu_deserialize(obj, location):
    if location == 'cpu':
        return obj


def _cuda_deserialize(obj, location):
    if location.startswith('cuda'):
        if location[5:] == '':
            device = 0
        else:
            device = max(int(location[5:]), 0)

        if not torch.cuda.is_available():
            raise RuntimeError('Attempting to deserialize object on a CUDA '
                               'device but torch.cuda.is_available() is False. '
                               'If you are running on a CPU-only machine, '
                               'please use torch.load with map_location=\'cpu\' '
                               'to map your storages to the CPU.')
        if device >= torch.cuda.device_count():
            raise RuntimeError('Attempting to deserialize object on CUDA device '
                               '{} but torch.cuda.device_count() is {}. Please use '
                               'torch.load with map_location to map your storages '
                               'to an existing device.'.format(
                                   device, torch.cuda.device_count()))
        return obj.cuda(device)


register_package(10, _cpu_tag, _cpu_deserialize)
register_package(20, _cuda_tag, _cuda_deserialize)

def load_bench(fn, map_location=None, pickle_module=pickle):
    new_fd = False
    if isinstance(fn, str) or \
            (sys.version_info[0] == 2 and isinstance(fn, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(fn, pathlib.Path)):
        new_fd = True
        f = open(fn, 'rb')
    try:
        return _load_bench(f,fn, map_location, pickle_module)
    finally:
        if new_fd:
            f.close()


def _load_bench(f, fn, map_location, pickle_module):
    deserialized_objects = {}
    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)
    elif isinstance(map_location, _string_classes):
        def restore_location(storage, location):
            return default_restore_location(storage, map_location)
    elif isinstance(map_location, torch.device):
        def restore_location(storage, location):
            return default_restore_location(storage, str(map_location))
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result



    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = inspect.getsource(container_type)
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn("Couldn't retrieve source code for container of "
                          "type " + container_type.__name__ + ". It won't be checked "
                          "for correctness upon loading.")
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + '.patch'
                diff = difflib.unified_diff(current_source.split('\n'),
                                            original_source.split('\n'),
                                            source_file,
                                            source_file, lineterm="")
                lines = '\n'.join(diff)
                try:
                    with open(file_name, 'a+') as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise IOError
                    msg = ("Saved a reverse patch to " + file_name + ". "
                           "Run `patch -p0 < " + file_name + "` to revert your "
                           "changes.")
                except IOError:
                    msg = ("Tried to save a patch, but couldn't create a "
                           "writable file " + file_name + ". Make sure it "
                           "doesn't exist and your working directory is "
                           "writable.")
            else:
                msg = ("you can retrieve the original source code by "
                       "accessing the object's source attribute or set "
                       "`torch.nn.Module.dump_patches = True` and use the "
                       "patch tool to revert the changes.")
            msg = ("source code of class '{}' has changed. {}"
                   .format(torch.typename(container_type), msg))
            warnings.warn(msg, SourceChangeWarning)
    def legacy_load(f):
        deserialized_objects = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
                mkdtemp() as tmpdir:
            print('opened tar file with fileobj')
            tar.extract('storages', path=tmpdir)
            #print ('opened tar file after storage extract')
            with open(os.path.join(tmpdir, 'storages'), 'rb', 0) as f:
                num_storages = pickle_module.load(f)
                for i in range(num_storages):
                    args = pickle_module.load(f)
                    key, location, storage_type = args
                    obj = storage_type._new_with_file(f)
                    obj = restore_location(obj, location)
                    deserialized_objects[key] = obj

                storage_views = pickle_module.load(f)
                for target_cdata, root_cdata, offset, size in storage_views:
                    root = deserialized_objects[root_cdata]
                    deserialized_objects[target_cdata] = root[offset:offset + size]
            print ('opened tar file after storage extract')
            tar.extract('tensors', path=tmpdir)
            #print ('opened tar file after tensor extract')
            with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
                num_tensors = pickle_module.load(f)
                for _ in range(num_tensors):
                    args = pickle_module.load(f)
                    key, storage_id, original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    tensor_type = storage_to_tensor_type(storage)
                    ndim, = struct.unpack('<i', f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    size = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    stride = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    storage_offset, = struct.unpack('<q', f.read(8))
                    tensor = tensor_type().set_(storage, storage_offset, size, stride)
                    deserialized_objects[key] = tensor
            print ('opened tar file after tensor extract')
            pickle_file = tar.extractfile('pickle')
            unpickler = pickle_module.Unpickler(pickle_file)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            print ('opened tar file after unpickler load')
            return result

    deserialized_objects = {}    

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        data = saved_id[1:]

        if typename == 'module':
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == 'storage':
            data_type, root_key, location, size, view_metadata = data
            if root_key not in deserialized_objects:
                deserialized_objects[root_key] = restore_location(
                    data_type(size), location)
            storage = deserialized_objects[root_key]
            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = storage[offset:offset + view_size]
                return deserialized_objects[view_key]
            else:
                return storage
        else:
            raise RuntimeError("Unknown saved id type: %s" % saved_id[0])
    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)
#    try:
#        t = tarfile.open(fn)
#        t.close()
#    except tarfile.TarError as e:
#        print ('open tar file with name error:',e)
#        pass
    if f_should_read_directly and f.tell() == 0:# and tarfile.is_tarfile(fn):
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            #print ('legacy_load:%s'%fn)
            return legacy_load(f)
        except tarfile.TarError:
            #print ('open tar file with fileobj error is:',e)
            # if not a tarfile, reset file offset and proceed
            f.seek(0)
    #print ('non legacy_load:%s'%fn)
    magic_number = pickle_module.load(f)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("Invalid protocol version: %s" % protocol_version)
    _sys_info = pickle_module.load(f)
    unpickler = pickle_module.Unpickler(f)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
    deserialized_storage_keys = pickle_module.load(f)
    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        deserialized_objects[key]._set_from_file(f, offset, f_should_read_directly)
        offset = None
    return result


if __name__== "__main__":
    if (len(sys.argv)!=3):
        print ('args: args: batch_size, number_buckets')
        exit()
    batch_size = int(sys.argv[1])
    num_bucket = int(sys.argv[2])
    #testf='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache/bucket_0/trace10014_len6.pt'
    #print (result.samples_observed)
    if (num_bucket <1 or batch_size <0 or batch_size > 12440):
        print ('buckets [1-10], batch_size [1,12440]')
        exit()
    rootdir='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache'
    trace_cache = []
    buckets_random =  random.sample(range(1,10), num_bucket)
    start=time.time()
    loaded=0
    for i in buckets_random:
        buckidx = i
        parent_dir = trace_cache_path = rootdir + '/bucket_' + str(buckidx)
        files = [name for name in os.listdir(parent_dir)]
        files = list(map(lambda f: os.path.join(parent_dir, f), files))
        if(batch_size>len(files)):
            print ('batch size:%d is larger than current bucket:%d files:%d'%(batch_size, i, len(files)))
            exit()
        chosen_files= random.sample(files, batch_size) # select batch_size files from current bucket
        for trace_file in chosen_files:
            try:
                new_trace = load_bench(trace_file)
                loaded+=1
                trace_cache.append(new_trace)
            except Exception as e:
                print ('loading error:')
        print ('done with bucket %d'%buckidx)
    end=time.time()
    print ('Loaded %d traces, %f bytes Time:%f seconds'%(loaded,sys.getsizeof(trace_cache), end-start))
    
    try:
      sample_t = random.sample(range(1,loaded),1)[0]
      print ('Sample trace:%d'%sample_t)
      sample = trace_cache[sample_t]
      print (sample)
      print (sample.samples_observed)
    except Exception as e:
      print (e)
