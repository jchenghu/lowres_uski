import os
import random
import numpy as np
import torch
import argparse
from argparse import Namespace
from time import time

import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data.vocab_tokenizer import SharedVocabTokenizer
from data.nmt import NmtDataSet, NmtBucketBatchDataSet, NmtBucketBatchCollator
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint
from loss.loss import LabelSmoothingLoss
from utils.args_utils import scheduler_type_choice
from test_withBPE import compute_mean_accuracy, count_parameters, \
    eval_compute_score_on_set_withBPE

# ----------------------- Global settings ------------------------------------

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
import functools
print = functools.partial(print, flush=True)

# --------------------------------------------------------------------

# prevents some issues related to torch low level implementation in some machines
# to be removed if they have no effect
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn', force=True)



def nmt_train(rank, world_size,
              nmt_train_ds, nmt_val_ds, nmt_test_ds,
              ddp_model, train_args, sched_args, save_args, other_args):
    num_workers = 0

    train_sampler = DistributedSampler(dataset=nmt_train_ds, seed=other_args.seed, drop_last=True,
                                       rank=rank, num_replicas=world_size)
    train_collator = NmtBucketBatchCollator(nmt_train_ds.get_src_vocab().get_pad_idx(), nmt_train_ds.get_trg_vocab().get_pad_idx())
    # batch size e' a 1, perche' dataset e' gia organizzato lui in batches, quindi fetchare 1, da' gia un [dyn_batch, seq_len]
    train_dl = DataLoader(nmt_train_ds, batch_size=1, num_workers=num_workers,
                          shuffle=False, collate_fn=train_collator, sampler=train_sampler)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), betas=(0.9, 0.98), eps=1e-9, lr=1.0)
    loss_function = LabelSmoothingLoss(smoothing_coeff=0.1,
                                       ignore_index=nmt_train_ds.get_trg_vocab().get_pad_idx(), rank=rank)
    loss_function.to(rank)

    if save_args.save_path is not None:
        found_checkpoint, _ = load_most_recent_checkpoint(ddp_model.module, optimizer, None,
                                                          rank, save_args.save_path)

    sched_it = 0
    algorithm_start_time = time()
    print("How many batches: " + str(len(train_dl)))
    for epoch in range(train_args.num_epochs):

        train_loss = 0
        train_acc = 0
        for batch, i in zip(iter(train_dl), range(len(train_dl))):
            ddp_model.train()

            # update iteration
            if sched_args.sched_type == 'noam':
                modified_sched_it = int(sched_it / train_args.num_accum)
                new_lr = pow(ddp_model.module.d_model, -0.5) * min(pow((modified_sched_it + 1), -0.5),
                            (modified_sched_it + 1) * pow(sched_args.warmup_steps, -1.5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            batch_src, num_pads_src, batch_trg, num_pads_trg = batch
            batch_input_x = batch_src.to(rank)
            batch_trg_y = batch_trg.to(rank)
            pred = ddp_model(enc_input=batch_input_x, dec_input=batch_trg_y[:, :-1],
                             enc_input_num_pads=num_pads_src, dec_input_num_pads=num_pads_trg,
                             apply_softmax=False)
            loss = loss_function(pred, batch_trg_y[:, 1:])
            loss.backward()

            if (i+1) % train_args.num_accum == 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            acc = compute_mean_accuracy(pred, batch_trg_y[:, 1:], num_pads_trg)
            if acc < 0:
                print("negative acc wtf...")
            train_acc += acc / len(pred)
            train_loss += loss.item()
            sched_it += 1

            del batch_src, num_pads_src, batch_trg, num_pads_trg
            torch.cuda.empty_cache()
        train_loss = train_loss / len(train_dl)
        train_acc = train_acc / len(train_dl)
        print("Epoch: " + str(epoch) + " Train loss: " + str(round(train_loss, 4)) + " acc: " + str(round(train_acc, 4)), end=" | ")

        # evaluation
        if rank == 0 and ((epoch+1) % 10 == 0) and epoch >= 200:

            print("Evaluation on Test Set")
            bleu_score = eval_compute_score_on_set_withBPE(ddp_model.module, nmt_test_ds,
                                                           beam_size=train_args.eval_beam_size,
                                                           trg_lang=train_args.trg_lang,
                                                           verbose=True)
            print("Test Bleu: " + str(bleu_score), end=" | ")

        print("lr: " + str(round(new_lr, 4)), end=" | ")
        print("Elapsed: " + str(round((time() - algorithm_start_time) / 60, 4)) + " min")


    save_last_checkpoint(ddp_model, optimizer, None,
                         save_args.save_path,
                         num_max_checkpoints=save_args.how_many_checkpoints)


def distributed_train(rank, world_size,
                      nmt_train_ds, nmt_val_ds, nmt_test_ds,
                      model_args, train_args, sched_args,
                      save_args, other_args):
    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = other_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print("Creating pretraining - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    if model_args.model_name == 'transformer':
        from model.transformer import TransformerMT
        model = TransformerMT(d_model=model_args.d_model, N=model_args.N, num_heads=1,
                              shared_word2idx=nmt_train_ds.get_src_word2idx_dict(),  # indifferente se prendo src o trg
                              d_ff=256, max_seq_len=model_args.max_len,
                              dropout_perc=model_args.dropout,
                              rank=rank)

    else:
        print("Must choose a model")
        exit(-1)
    print(model.__class__)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    print("The model has " + str(count_parameters(model)) + " trainable parameters")


    print("Start NMT Training")
    nmt_train(rank, world_size, nmt_train_ds, nmt_val_ds, nmt_test_ds,
              ddp_model, train_args, sched_args, save_args, other_args)

    dist.destroy_process_group()


def train(nmt_train_ds, nmt_val_ds, nmt_test_ds,
          model_args, train_args, sched_args,
          save_args, other_args):

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert (other_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(other_args.num_gpus))

    mp.spawn(distributed_train,
             args=(world_size,
                   nmt_train_ds, nmt_val_ds, nmt_test_ds,
                   model_args, train_args, sched_args,
                   save_args, other_args
                   ),
             nprocs=other_args.num_gpus,
             join=True)



if __name__ == "__main__":

    print("Starting: " + str(__name__))

    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('--selected_model', type=str, default='transformer')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    # scheduler arguments that are used or not according to the scheduler type
    parser.add_argument('--sched_type', type=scheduler_type_choice, default='noam')
    parser.add_argument('--warmup_steps', type=int, default=400)
    parser.add_argument('--pretrain_eps', type=float, default=1e-4)


    # train_hdf5_file=train_args.hdf5_path,
    # hdf5_num_batches=train_args.num_hdf5,

    parser.add_argument('--min_len', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--max_train_len', type=int, default=200)

    from utils.args_utils import str2bool
    parser.add_argument('--reverse_src_trg', type=str2bool, default=False)
    parser.add_argument('--ALL_INDEXES', type=str2bool, default=False)

    parser.add_argument('--print_every_iter', type=int, default=500)
    parser.add_argument('--num_pretrain_iter', type=int, default=15000)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--eval_batch_size', type=int, default=96)
    parser.add_argument('--pretrain_batch_size', type=int, default=96)
    parser.add_argument('--num_accum', type=int, default=2)
    parser.add_argument('--pretrain_num_accum', type=int, default=4)
    parser.add_argument('--eval_beam_size', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves/')
    parser.add_argument('--parent_path', type=str, default='')
    parser.add_argument('--how_many_checkpoints', type=int, default=1)

    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--language', type=str, default='kk-en')
    parser.add_argument('--eval_stats', type=str2bool, default=True)

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--ddp_sync_port', type=int, default=12354)

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    seed = args.seed
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if args.language == 'uz-en_4000':
        data_args = Namespace(data_path='./bpe_data/qed_uz_en/',
                              src_lang='uz',
                              trg_lang='en',
                              src_train_filename='4000_train.uz',
                              trg_train_filename='4000_train.en',
                              src_val_filename='4000_val.uz',
                              trg_val_filename='4000_val.en',
                              src_test_filename='4000_test.uz',
                              trg_test_filename='4000_test.en')
    else:
        print("Please choose a language pair.")
        raise ValueError

    model_args = Namespace(model_name=args.selected_model,
                           N=args.N,
                           d_model=args.d_model,
                           max_len=args.max_len,
                           dropout=args.dropout)
    train_args = Namespace(batch_size=args.batch_size,
                           pretrain_batch_size=args.pretrain_batch_size,
                           num_accum=args.num_accum,
                           pretrain_num_accum=args.num_accum,
                           num_epochs=args.num_epochs,
                           num_pretrain_iter=args.num_pretrain_iter,
                           print_every_iter=args.print_every_iter,
                           eval_batch_size=args.eval_batch_size,
                           eval_beam_size=args.eval_beam_size,
                           src_lang=data_args.src_lang,
                           trg_lang=data_args.trg_lang,
                           pretrain_eps=args.pretrain_eps,
                           eval_stats=args.eval_stats)
    sched_args = Namespace(sched_type=args.sched_type,
                           warmup_steps=args.warmup_steps)
    save_args = Namespace(save_path=args.save_path,
                          how_many_checkpoints=args.how_many_checkpoints)
    other_args = Namespace(ddp_sync_port=args.ddp_sync_port,
                           num_gpus=args.num_gpus,
                           seed=args.seed)

    shared_vocab = SharedVocabTokenizer(sentences_path_src=data_args.data_path + data_args.src_train_filename,
                                        sentences_path_trg=data_args.data_path + data_args.trg_train_filename,
                                        verbose=True)
    src_vocab = shared_vocab  # trick per usarne uno solo...
    trg_vocab = shared_vocab

    if args.reverse_src_trg:
        print("Swapping SRC and TRG languages")
        tmp = data_args.trg_train_filename
        data_args.trg_train_filename = data_args.src_train_filename
        data_args.src_train_filename = tmp

        tmp = data_args.trg_val_filename
        data_args.trg_val_filename = data_args.src_val_filename
        data_args.src_val_filename = tmp

        tmp = data_args.trg_test_filename
        data_args.trg_test_filename = data_args.src_test_filename
        data_args.src_test_filename = tmp

    nmt_val_ds = NmtDataSet(src_sentences_path=data_args.data_path + data_args.src_val_filename,
                            trg_sentences_path=data_args.data_path + data_args.trg_val_filename,
                            min_length=args.min_len, max_length=args.max_len,
                            src_vocab=src_vocab, trg_vocab=trg_vocab, verbose=True)
    nmt_test_ds = NmtDataSet(src_sentences_path=data_args.data_path + data_args.src_test_filename,
                             trg_sentences_path=data_args.data_path + data_args.trg_test_filename,
                             min_length=args.min_len, max_length=args.max_len,
                             tokenize_trg=False,
                             src_vocab=src_vocab, trg_vocab=trg_vocab, verbose=True)
    nmt_train_ds = NmtBucketBatchDataSet(src_sentences_path=data_args.data_path + data_args.src_train_filename,
                                         trg_sentences_path=data_args.data_path + data_args.trg_train_filename,
                                         min_length=args.min_len, max_length=args.max_train_len,
                                         bucket_size=train_args.batch_size,
                                         src_vocab=src_vocab, trg_vocab=trg_vocab, verbose=True)

    train(nmt_train_ds, nmt_val_ds, nmt_test_ds,
          model_args, train_args, sched_args, save_args, other_args)
