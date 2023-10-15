## Convert weights to GGML format

This subdirectory contains all the utilities necessary to download and convert BioGPT
weights into `ggml` format.

### Download weights

Run the following commands to download the weights for BioGPT:

```bash
./download_weights.sh
```

### Convert weights to ggml format

Once downloaded, the weights should be accessible in the `weights` directory at the root
of the repository. Run the command:

```bash
python convert_pt_to_ggml.py --dir-model ./weights/Pre-trained-BioGPT/ --out-dir ./ggml_weights
```

This creates a `ggml_weights` directory with a file containing the weights stored in binary
in a ggml-friendly format.

You are now ready to use the `biogpt.cpp` library.
