for SPLIT in train val
do
  for LANG in source target
  do
    python -m multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "subtaskC_data/$SPLIT.$LANG" \
    --outputs "subtaskC_data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done