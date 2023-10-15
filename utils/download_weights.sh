#!/bin/bash

WEIGHTS_DIR="../weights"

WEIGHTS_LINKS=(
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/Pre-trained-BioGPT.tgz"
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/Pre-trained-BioGPT-Large.tgz"
    "https://msralaphilly2.blob.core.windows.net/release/BioGPT/checkpoints/QA-PubMedQA-BioGPT.tgz"
    "https://msralaphilly2.blob.core.windows.net/release/BioGPT/checkpoints/QA-PubMedQA-BioGPT-Large.tgz"
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/RE-BC5CDR-BioGPT.tgz"
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/RE-DDI-BioGPT.tgz"
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/RE-DTI-BioGPT.tgz"
    "https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/DC-HoC-BioGPT.tgz"
)

if [ ! -d "$WEIGHTS_DIR" ]; then
    mkdir -p "$WEIGHTS_DIR"
    echo "weights directory created."
else
    echo "weights already exists."
fi

for link in "${WEIGHTS_LINKS[@]}"; do
    filename=$(basename "$link")

    if [ -f "$filename" ]; then
        echo "$filename already exists. Skipping download."
    else
        wget "$link" -P "$WEIGHTS_DIR"
        tar -zxvf "${WEIGHTS_DIR}/${filename}.tgz"
        rm "${WEIGHTS_DIR}/${filename}.tgz"
    fi
done
