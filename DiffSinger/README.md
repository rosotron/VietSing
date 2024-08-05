# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism


## DiffSinger

The pipeline below is designed for VietSing dataset:

### 1. Preparation

#### Data Preparation
a) Download and extract VietSing, then add to the dataset folder: `data/raw/vietsing`

b) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
# For every new dataset, run data_gen/tts/help.py and change spec_min and spec_max in usr/configs/midi/cascade/xxx/xxx_statis.yaml
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/midi/cascade/vietsing/viet_aux_rel.yaml

# `data/binary/vietsing-bin` will be generated.
```

#### Vocoder Preparation
We use the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip).

Also, please unzip pre-trained vocoder and [this pendant for vocoder](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0102_xiaoma_pe.zip) into `checkpoints` before training your acoustic model.

You can also move [a ckpt with more training steps](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/model_ckpt_steps_1512000.ckpt) into this vocoder directory

This singing vocoder is trained on ~70 hours singing data, which can be viewed as a universal vocoder. 

#### Exp Name Preparation
```bash
export MY_DS_EXP_NAME=0804_vietsing_ds100_rel
```

```
.
|--data
    |--raw
        |--vietsing
            |--segments
                |--transcriptions.txt
                |--wavs
|--checkpoints
    |--MY_DS_EXP_NAME (optional)
    |--0109_hifigan_bigpopcs_hop128 (vocoder)
        |--model_ckpt_steps_1512000.ckpt
        |--config.yaml
```

### 2. Training Example
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/vietsing/viet_ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset  
```

### 3. Inference from packed test set
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/vietsing/viet_ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset --infer
```
Inference results will be saved in `./checkpoints/MY_DS_EXP_NAME/generated_` by default.

We also provide:
 - the pre-trained model of DiffSinger;
 
They can be found in [here](https://drive.google.com/file/d/1wYAIVSDX1Ra5FXa7wsRQ6cZMUR_gK92y/view?usp=sharing).

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Inference from raw inputs
```sh
python inference/svs/ds_e2e.py --config usr/configs/midi/e2e/vietsing/viet_ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME
```

