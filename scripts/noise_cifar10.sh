#!/bin/bash

# custom config
DATA="DATA/"
MODEL=fedavg
TRAINER=PromptFL
PRETRAINED=True
NUM_USERS=2
train_batch_size=2
test_batch_size=2
LR=0.001
GAMMA=1
USERS=10
FRAC=1
ROUND=2
NUM_PROMPT=2
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CTXINIT=False
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
BETA=0.1
# SEED=1
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
for DATASET in cifar10
do
  for PARTITION in noniid-labeldir
  do
    for SEED in 1
    do
      DIR=output/${DATASET}_${PARTITION}_beta${BETA}/${MODEL}_${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}_noisy
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python federated_main.py \
        --root ${DATA} \
        --model ${MODEL} \
        --seed ${SEED} \
        --num_users ${USERS} \
        --frac ${FRAC} \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --n_ctx ${NCTX} \
        --num_prompt ${NUM_PROMPT} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/PromptFL/vit_b16.yaml \
        --output-dir ${DIR}
      fi
    done
  done
done
