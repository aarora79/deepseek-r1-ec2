#!/bin/sh

################################################################################
# Script Name: deploy_model.sh
# Description: Deploys a vLLM model in a Docker container with configurable parameters
# Author: Amit Arora
# Date: 2025-01-27
#
# Usage: ./deploy_model.sh [model_id] [tp_degree]
# Example: ./deploy_model.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B 4
################################################################################

# Exit immediately if any command fails
set -e

#------------------------------------------------------------------------------
# Environment variable validation
#------------------------------------------------------------------------------
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set it using: export HF_TOKEN=your_token"
    exit 1
fi

#------------------------------------------------------------------------------
# Configuration parameters with defaults
#------------------------------------------------------------------------------
# Model identifier from HuggingFace
MODEL_ID="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
# Tensor parallel degree for model deployment
TP_DEGREE="${2:-4}"
# Container name constant
CONTAINER_NAME="fmbench_model_container"
# Maximum attempts for container cleanup
MAX_CLEANUP_ATTEMPTS=3
# Wait time between cleanup attempts (in seconds)
CLEANUP_WAIT_TIME=5

#------------------------------------------------------------------------------
# Container cleanup function
#------------------------------------------------------------------------------
cleanup_container() {
    local container_exists
    container_exists=$(docker ps -aq --filter "name=$CONTAINER_NAME")

    if [ -n "$container_exists" ]; then
        for i in $(seq 1 $MAX_CLEANUP_ATTEMPTS); do
            echo "Attempt $i to stop and remove the container: $CONTAINER_NAME"

            # Stop the container if running
            docker ps -q --filter "name=$CONTAINER_NAME" | xargs -r docker stop

            echo "Waiting $CLEANUP_WAIT_TIME seconds for container to stop..."
            sleep $CLEANUP_WAIT_TIME

            # Remove the container
            docker ps -aq --filter "name=$CONTAINER_NAME" | xargs -r docker rm

            echo "Waiting $CLEANUP_WAIT_TIME seconds to verify removal..."
            sleep $CLEANUP_WAIT_TIME

            # Check if container was successfully removed
            if [ -z "$(docker ps -aq --filter "name=$CONTAINER_NAME")" ]; then
                echo "Container $CONTAINER_NAME successfully stopped and removed."
                return 0
            else
                echo "Container $CONTAINER_NAME still exists, retrying..."
            fi
        done
        echo "Warning: Failed to remove container after $MAX_CLEANUP_ATTEMPTS attempts"
        return 1
    else
        echo "Container $CONTAINER_NAME does not exist. No cleanup needed."
        return 0
    fi
}

#------------------------------------------------------------------------------
# Main execution
#------------------------------------------------------------------------------
echo "Starting deployment process..."
echo "Model ID: $MODEL_ID"
echo "Tensor Parallel Degree: $TP_DEGREE"

# Perform container cleanup
cleanup_container

# Deploy new container
echo "Deploying new container..."
docker run -d \
    --rm \
    --name="$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    -p 8000:8000 \
    vllm/vllm-openai:v0.7.1 \
    --model "$MODEL_ID" \
    --tensor-parallel-size "$TP_DEGREE" \
    --max-model-len 32768 \
    --enforce-eager

echo "Container deployment completed successfully"
echo "Service should be available on port 8000"
