python run.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_train --train_data_file=../dataset/cdata/nobalance/train_cdata.jsonl --eval_data_file=../dataset/cdata/nobalance/valid_cdata.jsonl --test_data_file=../dataset/cdata/nobalance/test_cdata.jsonl --epoch 8 --block_size 400 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 --pkl_file=short_3path_cdata_nobalance.pkl
python run.py --output_dir=./saved_models --model_type=roberta --tokenizer_name=../models/codebert --model_name_or_path=../models/codebert --do_eval --do_test --train_data_file=../dataset/cdata/nobalance/train_cdata.jsonl --eval_data_file=../dataset/cdata/nobalance/valid_cdata.jsonl --test_data_file=../dataset/cdata/nobalance/test_cdata.jsonl --epoch 8 --block_size 400 --train_batch_size 40 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 --pkl_file=short_3path_cdata_nobalance.pkl
python ../evaluator/evaluator.py -a ../dataset/cdata/nobalance/test_cdata.jsonl -p saved_models/predictions.txt