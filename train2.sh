MAX_UPDATES=12000      # Number of training steps.
WARMUP_UPDATES=500    # Linearly increase LR over this many steps.
LR=1e-05              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=2     # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=./roberta.large/model.pt
DATA_DIR=./data

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:

FAIRSEQ_USER_DIR=.

CUDA_VISIBLE_DEVICES=0,3 fairseq-train --fp16 --memory-efficient-fp16 --ddp-backend=no_c10d \
    $DATA_DIR \
    --save-dir ./checkpoints/taskB \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task commonsense_ve2 --init-token 0 --bpe gpt2 \
    --arch roberta_large --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 3 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --seed $SEED \
    --log-format simple --log-interval 250 2>&1 | tee train.log


# --find-unused-parameters \ --memory-efficient-fp16
# --num-workers=0 \ 
#false only : best 0.93
#bz 4 : best window3    0.936
#memory efficient : best window2    0.937
#memory efficient + true : window1  0.937
#memory efficient + true + wiki bz 6: 0.931 (epoch 4)


    
    #win0 true  win3 wik   win4 true+wik