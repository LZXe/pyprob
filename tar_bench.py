import cProfile
import tarfile
import time
def taropen_with_filename(filename):
    try: 
       t = tarfile.open(filename)
       t.close()
    except tarfile.TarError as e:
       print ('tarError with name:',e)
       pass


def taropen_with_fileobj(f):
    try:
        tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)
    except tarfile.TarError as e:
        print ('tarError with obj:',e) 

#testdir='/global/cscratch1/sd/jialin/etalumis_data/etalumis_data_july30/trace_cache/'
testdir='/global/homes/j/jialin/pyprob_io/pyprob/'
testfile1='trace8643_len7.pt'
#testfile1='trace10335_len7.pt'
testfile2='trace8643_len7_copy.pt'
#trace10337_len7.pt
#trace1033_len7.pt'

testfile=testdir+testfile1
name_s=time.time()
taropen_with_filename(testfile)
name_e=time.time()  

testfile=testdir+testfile2
f=open(testfile,'rb')
obj_s=time.time()
taropen_with_fileobj(f)
obj_e=time.time()


print ('filename:%f'%(name_e-name_s))
print ('fileobj :%f'%(obj_e-obj_s))
