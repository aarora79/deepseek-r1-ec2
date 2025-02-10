#!/bin/bash

# Default values
MODEL_PATH=""
CACHE_TYPE="q4_0"
THREADS=48
PRIO=3
TEMP=0.6
CTX_SIZE=8192
SEED=3407
GPU_LAYERS=62
BATCH_SIZE=4096
NUMA_OPTION="distribute"
MEMORY_THRESHOLD=95

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model         Path to the model file (REQUIRED)"
    echo "  -c, --cache-type    Cache type (default: q4_0)"
    echo "  -t, --threads       Number of threads (default: 48)"
    echo "  -p, --prio          Priority (default: 3)"
    echo "  --temp              Temperature (default: 0.6)"
    echo "  --ctx-size          Context size (default: 8192)"
    echo "  --seed              Random seed (default: 3407)"
    echo "  --gpu-layers        Number of GPU layers (default: 62)"
    echo "  --batch-size        Batch size (default: 4096)"
    echo "  --numa              NUMA option (default: distribute)"
    echo "  --mem-threshold     Memory threshold % (default: 95)"
    exit 1
}

# Parse command-line arguments
PARSED_ARGUMENTS=$(getopt -a -n run_llama_server \
    -o m:c:t:p: \
    --long model:,cache-type:,threads:,prio:,temp:,ctx-size:,seed:,gpu-layers:,batch-size:,numa:,mem-threshold: \
    -- "$@")
VALID_ARGUMENTS=$?
[ $VALID_ARGUMENTS -ne 0 ] && usage

eval set -- "$PARSED_ARGUMENTS"
while :
do
    case "$1" in
        -m | --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c | --cache-type)
            CACHE_TYPE="$2"
            shift 2
            ;;
        -t | --threads)
            THREADS="$2"
            shift 2
            ;;
        -p | --prio)
            PRIO="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --numa)
            NUMA_OPTION="$2"
            shift 2
            ;;
        --mem-threshold)
            MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --) 
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required"
    usage
fi

# Construct Llama server command
LLAMA_SERVER_CMD="${HOME}/llama.cpp/llama-server \
    --model ${MODEL_PATH} \
    --cache-type-k ${CACHE_TYPE} \
    --threads ${THREADS} \
    --prio ${PRIO} \
    --temp ${TEMP} \
    --ctx-size ${CTX_SIZE} \
    --seed ${SEED} \
    --n-gpu-layers ${GPU_LAYERS} \
    --batch-size ${BATCH_SIZE} \
    --numa ${NUMA_OPTION}"

# Function to check CUDA memory for multi-GPU systems
check_cuda_memory() {
    # Get detailed GPU memory information
    local nvidia_output=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    
    # Convert output to array
    IFS=$'\n' read -d '' -r -a memory_lines <<< "$nvidia_output"
    
    local max_memory_percentage=0
    
    # Calculate memory percentage for each GPU
    for line in "${memory_lines[@]}"; do
        # Split line into used and total memory
        IFS=', ' read -r used total <<< "$line"
        
        # Calculate memory percentage
        local memory_percentage=$(( (used * 100) / total ))
        
        # Track the highest memory percentage
        if (( memory_percentage > max_memory_percentage )); then
            max_memory_percentage=$memory_percentage
        fi
    done
    
    # Check if any GPU is above the threshold
    if [[ $max_memory_percentage -ge $MEMORY_THRESHOLD ]]; then
        echo "CUDA memory critically low: ${max_memory_percentage}% used on at least one GPU"
        return 1
    fi
    return 0
}

# Function to find and kill existing llama-server
kill_existing_server() {
    pkill -f "llama-server"
    sleep 2  # Give some time for process to terminate
}

# Function to start llama-server
start_server() {
    echo "Starting llama-server..."
    echo "Command: $LLAMA_SERVER_CMD"
    nohup $LLAMA_SERVER_CMD > /dev/null 2>&1 &
    sleep 5  # Give some time for server to start
}

# Main monitoring loop
while true; do
    # Check if llama-server is running
    if ! pgrep -f "llama-server" > /dev/null; then
        echo "Llama server not running. Starting server..."
        start_server
    fi

    # Check CUDA memory
    if ! check_cuda_memory; then
        echo "CUDA memory critical. Restarting server..."
        kill_existing_server
        start_server
    fi

    # Wait for 1 second before next check
    sleep 1
done
