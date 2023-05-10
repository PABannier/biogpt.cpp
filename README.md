# Step 1: Download the weights of BioGPT

The network is trained on different tasks, all weights should be downloaded.
Output: a bash script downloading all the weights from the server.


# Step 2: Parse the weights into the GGML format

Write a script to load the torch checkpoints and convert the checkpoints into the
GGML format.
Output: a Python script that parses all the checkpoints.


# Step 3: Actual implementation of the BioGPT with GGML

Loading the tensor weights into memory.
Output: main.cpp loading the weights.

# Step 4: Example script 