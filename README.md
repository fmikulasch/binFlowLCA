# Binocular flow field LCA

## Contents of this folder

simulation/ - Files for creating optical flow fields
input/ - Input files the network will use
output/ - Output files the network produces
script/ - Scripts for analyzing the output files
analysis/ - Products of analysis should be saved here

## Creating flow fields

- *Start*-script in the *sim.blend* creates sequences of images
-  *flowCalculation.py* creates the flow fields

## Running the network

- First convert the *.lua*: `lua bin_motion_LCA.lua > bin_motion_LCA.params`
- Start the network: `mpirun -np 1 ../../build/tests/BasicSystemTest/Release/BasicSystemTest -p bin_motion_LCA.params -batchwidth 1 -l BinFlow.log -t 8`
