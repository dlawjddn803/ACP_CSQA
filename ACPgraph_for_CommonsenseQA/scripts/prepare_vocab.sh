dataset=./amr_data/amr_2.0/csqa
python3 extract_vocab.py --train_data ${dataset}/train.pred.txt \
                   --amr_files ${dataset}/train.pred.txt ${dataset}/dev.pred.txt ${dataset}/test.pred.txt \
                   --nprocessors 8
mv *_vocab ${dataset}/.
