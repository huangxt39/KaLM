fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "subtaskC_data/train.bpe" \
  --validpref "subtaskC_data/val.bpe" \
  --destdir "subtaskC_data-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;