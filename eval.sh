export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# IF the GPUS are 4000 series, use the following command
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node 1 src/eval.py pretrained_model_path="/home/zzzzz/vsr_llm_lightning/ckpts/2024-11-24_17-58-54/last.ckpt" >eval.log 2>&1