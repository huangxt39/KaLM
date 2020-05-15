MAX_UPDATES=8000      # Number of training steps.
WARMUP_UPDATES=800    # Linearly increase LR over this many steps.
LR=1e-5              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=4      # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=./roberta.large/model.pt
DATA_DIR=./data

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:

FAIRSEQ_USER_DIR=.
#--no-last-checkpoints --no-epoch-checkpoints
CUDA_VISIBLE_DEVICES=4,5 fairseq-train --fp16 --memory-efficient-fp16 --ddp-backend=no_c10d \
    $DATA_DIR \
    --save-dir ./checkpoints/taskA \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task commonsense_ve --init-token 0 --bpe gpt2 \
    --arch roberta_large --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --seed $SEED \
    --disable-validation \
    --log-format simple --log-interval 250 2>&1 | tee train.log
    
#0.952

#sent only capitalized 0.957
#sent only no capit    0.9544
#sent+evidence cap     0.963    ->test on test split   1: 91.0   2:93.3  3:93.7   4:94.1   5:94.0