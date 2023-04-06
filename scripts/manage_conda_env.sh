#!/bin/bash


DEFAULT_ENV_NAME="gpt-cache"

# Usage: ./manage_conda_env.sh create [env_name]
# Usage: ./manage_conda_env.sh remove [env_name]

if [[ "$1" == "create" ]]; then
    if [[ -n "$2" ]]; then
        env_name="$2"
    else
        env_name="$DEFAULT_ENV_NAME"
    fi
    if conda env list | grep -q "^$env_name "; then
        echo "conda environment '$env_name' already exists."
    else
        conda create --name "$env_name" python=3.8
        echo "conda environment '$env_name' created."
    fi
    conda activate "$env_name"
    echo "conda environment '$env_name' activated."
elif [[ "$1" == "remove" ]]; then
    conda deactivate
    if [[ -n "$2" ]]; then
        env_name="$2"
    else
        env_name="$DEFAULT_ENV_NAME"
    fi
    conda remove --name "$env_name" --all
    echo "conda environment '$env_name' removed."
else
    echo "Usage: ./manage_conda_env.sh [create|remove] [env_name]"
    exit 1
fi