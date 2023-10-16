# Configure your training (DDP? fp16? ...)
# see https://huggingface.co/docs/accelerate/index for details
# accelerate config

# Where your data is
DATA_DIR=data/outdoor/YangzhouSign_Dawn_Nonref_Insta_360/data/outputs/
EXP_NAME=Yangzhou_sign

# Experiment will be conducted under "exp/${EXP_NAME}" folder
# "--gin_configs=configs/360.gin" can be seen as a default config
# and you can add specific config useing --gin_bindings="..."
#accelerate launch train.py \
#    --gin_configs=configs/359.gin \
#    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
#    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
#    --gin_bindings="Config.factor = 3"

# or you can also run without accelerate (without DDP)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXP_NAME}'" \
      --gin_bindings="Config.factor = 4"

# alternatively you can use an example training script
#bash scripts/train_360.sh

# blender dataset
#bash scripts/train_blender.sh

# metric, render image, etc can be viewed through tensorboard
#tensorboard --logdir "exp/${EXP_NAME}"
