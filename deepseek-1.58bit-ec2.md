## How to run DeepSeek 1.58bit Quant On g6e.12xlarge


**Launch a g6e.12xlarge instance with atleast 450GB Storage with 16000 IOPS and Throughput to be 1000.**

## Start by cloning and building llama.cpp

```{bash}

apt-get update
apt-get install build-essential cmake curl libcurl4-openssl-dev -y
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-server
cp llama.cpp/build/bin/llama-* llama.cpp

```

## Download the snapshot of Deepseek R1 from Huggingface

```{bash}

pip install hf_transfer
pip install huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1

```

```{bash}


huggingface-cli download unsloth/DeepSeek-R1-GGUF \
  --local-dir DeepSeek-R1-GGUF \
  --include "*UD-IQ1_S*"

```

## Then run llama-cli, If you want to serve this, then use the second command.

```{bash}
./llama.cpp/llama-cli \
    --model /home/ubuntu/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
    --cache-type-k q4_0 \
    --threads 16 \
    --prio 2 \
    --temp 0.6 \
    --ctx-size 8192 \
    --seed 3407 \
    --n-gpu-layers 16 \
    -no-cnv \
    --prompt "<｜User｜>Create a Flappy Bird game in Python.<｜Assistant｜>"
```


```{bash}

./llama.cpp/llama-server \
  --model /home/ubuntu/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
  --cache-type-k q4_0 \
  --threads 48 \ # use all threads
  --prio 3 \
  --temp 0.6 \
  --ctx-size 8192 \
  --seed 3407 \
  --n-gpu-layers 62 # all layers are offloaded to the GPUs
  -np 4 
```
