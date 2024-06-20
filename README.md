### Learning from Wrong Predictions in Low-Resource Neural Machine Translation

Clean and provvisory code base for USKI pre-training described in "[Learning from Wrong Predictions in Low-Resource Neural Machine Translation](https://aclanthology.org/2024.lrec-main.896/)" presented at [LREC-COLING 2024](https://lrec-coling-2024.org/).<br>

##### Requirements
* python >= 3.7
* torch

#### Sample Usage

The dataset Uzbek-English translation dataset is selected as an example.

USKI Pre-training

``` 
python3.7 train_USKI.py --selected_model transformer --N 3 \
    --dropout 0.15 --seed 1234  --num_epochs 300 \
    --max_len 250  --min_len 2 \
    --d_model 128 \
    --sched_type noam --warmup_steps 400 \
    --batch_size 2048 --num_accum 2 \
    --pretrain_batch_size 128 --pretrain_num_accum 2 \
    --num_pretrain_iter 42000 \
    --eval_beam_size 4 \
    --eval_batch_size 32 \
    --save_path ./github_ignore_material/saves/ \
    --language uz-en_4000 --num_gpus 1 \
    --ddp_sync_port 12139 &> output.txt &
```

Training without USKI

``` 
python3.7 train.py \
            --selected_model transformer --N 3 \
            --dropout 0.15 --seed 1234  --num_epochs 300 \
            --max_len 200  --min_len 2 \
            --d_model 128 \
            --sched_type noam --warmup_steps 400 \
            --batch_size 2048 --num_accum 2 \
            --eval_beam_size 4 \
            --eval_batch_size 32 \
            --save_path ./github_ignore_material/saves/ \
            --language uz-en_4000 --num_gpus 1 \
            --ddp_sync_port 12139 &> NORMAL_output.txt &

```

Evaluation is based on SacreBLEU and it is automatically handled. <br>

Tune the arguments "batch size", "num_accum", "pretrain_batch_size" and "pretrain_num_accum" 
according to the available computational resources. They refer to the batch size
in the standard training and during the USKI pre-training. <br><br>

Visualization is provided on temrinal, results can be monitored in the files  `NORMAL_output.txt` and `NORMAL_output.txt`.


##### Reference

If you find this repository useful, please consider citing (no obligation):

```
@inproceedings{hu2024learning,
  title={Learning from Wrong Predictions in Low-Resource Neural Machine Translation},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Berardinelli, Giulia and Capotondi, Alessandro},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={10263--10273},
  year={2024}
}
```

#### Acknowledgments

We thank the contributors of the [OPUS Corpus](https://opus.nlpl.eu/) which provided
the Uzbek-English dataset. <br>
We also adopted the [BPE](https://github.com/rsennrich/subword-nmt) tokenization algorithm. <br>


