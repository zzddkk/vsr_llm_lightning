export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
# IF the GPUS are 4000 series, use the following command
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node 2 src/train.py >train.log 2>&1