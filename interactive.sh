MODEL_DIR=checkpoints/taskC
fairseq-interactive \
    --path $MODEL_DIR/checkpoint_best.pt preprocess_taskC_data/subtaskC_data-bin \
    --beam 5 --source-lang source --target-lang target \
    --tokenizer space \
    --bpe gpt2 --gpt2-encoder-json preprocess_taskC_data/encoder.json --gpt2-vocab-bpe preprocess_taskC_data/vocab.bpe