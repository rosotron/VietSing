
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

CUDA_VISIBLE_DEVICES=0 python inference.py \
    -model_dir ./checkpoints \
    -input_dir ./data/openviet/testset.txt \
    -output_dir ./results \

