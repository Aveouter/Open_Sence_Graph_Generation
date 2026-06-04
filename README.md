# 验证
python train.py --test \
  --method RelTR \
  --dataname VisualGenome \
  --ckpt_path results/ckpts/checkpoint0149.pth \
  --gpus 0 2 5 6
