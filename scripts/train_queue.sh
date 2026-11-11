#!/usr/bin/env bash
# Train a list of models one after the other on one GPU.
# Usage: scripts/train_queue.sh <gpu> <arch>:<encoder> [<arch>:<encoder> ...]
set -u
gpu=$1; shift
logdir=checkpoints/OpticDisk-Macula-Segmentation-v2/logs
mkdir -p "$logdir"
for model in "$@"; do
    arch=${model%%:*}; encoder=${model#*:}
    echo "$(date '+%F %T') START $arch $encoder on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py --arch "$arch" --encoder "$encoder" \
        > "$logdir/$arch-$encoder.log" 2>&1
    echo "$(date '+%F %T') END $arch $encoder exit=$?"
done
