#!/bin/bash
module load python/3.6-anaconda-5.2
source activate conda_notebook
plasma_store -m 10000000000 -s /tmp/plasma &

echo 'plasma store started with 10 GB memory'
