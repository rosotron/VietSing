# VISinger2

You can refer to the official repo of [VISinger2](https://github.com/zhangyongmao/VISinger2) and follow the same steps.

## Pre-requisites
1. Install python requirements: pip install -r requirements_3090.txt
2. Download the VietSing dataset.
3. prepare data like data/vietsing (wavs, trainset.txt, testset.txt, train.list, test.list)
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
The pre-trained model can be found [here]()
