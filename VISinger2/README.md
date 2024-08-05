# VISinger2

Please refer to the official repo of [VISinger2](https://github.com/zhangyongmao/VISinger2) and follow the same steps.

This repository is the official PyTorch implementation of [VISinger2](https://arxiv.org/abs/2211.02903).

### Updates
- Apr 10 2023: Add egs/visinger2_flow: add flow to VISinger2 to get a more flexible prior distribution.
- Jan 31 2023: Modify the extraction method of gt-dur in dataset.py. Replace the dsp-wav with a sinusoidal signal as input to the HiFi-GAN decoder.
- Jan 10 2023: Init commit.

## Pre-requisites
1. Install python requirements: pip install -r requirements_3090.txt
2. Download the VietSing dataset.
3. prepare data like data/opencpop (wavs, trainset.txt, testset.txt, train.list, test.list)
4. modify the egs/visinger2/config.json (data/data_dir, train/save_dir)

## extract pitch and mel
```
cd egs/visinger2
bash bash/preprocess.sh config.json
```

## Training
```
cd egs/visinger2
bash bash/train.sh 0
```

## Inference
modify the model_dir, input_dir, output_dir in inference.sh
```
cd egs/visinger2
bash bash/inference.sh
```

