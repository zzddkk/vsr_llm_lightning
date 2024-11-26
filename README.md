## Set the env
```
conda create -n vsr_llm python=3.10.12 -y
conda install ffmpeg==7.1.0
python -m pip install --upgrade pip==23.2 (from 24.3.1)
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
pip install -r ./requirements.txt
pip install hydra-core --upgrade (it will occur error but ignore)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## [preparation](https://github.com/mpc001/auto_avsr/tree/main/preparation)
1. process the dataset from auto_avsr
2. rename the labels to label or alter the parmas in datamoudule or dataset

## Note 
1. The cfg.dataset.(max_frams max_val_frams) is not for total_gpus only for one.Per 4090 I suggest the value is (1200,1200) A100 (1800,1800)
2. the test.py is only support 1 gpu
3. when you want to eval the model (pretrain_model_path or resume_model_path) can use