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
import torch.nn as nn

from data.pretraining import Pretrain_USKI_Dataset, Pretrain_USKI_Collator
from data.pretraining_Selective import PretrainSelectiveCollator, PretrainSelectiveDataset

from data.vocab_tokenizer import SharedVocabTokenizer
from data.nmt import NmtDataSet, NmtCollator, NmtBucketBatchDataSet, NmtBucketBatchCollator
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint
from loss.loss import LabelSmoothingLoss
from utils.args_utils import scheduler_type_choice
from test_withBPE import compute_mean_accuracy, \
    eval_compute_score_on_set_withBPE

# ----------------------- Global settings ------------------------------------

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
import functools
print = functools.partial(print, flush=True)

# --------------------------------------------------------------------

# solves weird bugs with torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_start_method('spawn', force=True)


def pretrain_USKI(rank, world_size,

                   train_src_sentences_tokenized, train_trg_sentences_tokenized,

                   nmt_val_ds,

                   pretrain_train_ds, pretrain_val_ds,
                   ddp_model, train_args, sched_args, save_args, other_args):
    num_workers = 0
    train_sampler = DistributedSampler(dataset=pretrain_train_ds, seed=other_args.seed, drop_last=True,
                                       rank=rank, num_replicas=world_size)
    train_collator = PretrainSelectiveCollator(
                                           train_src_sentences_tokenized, train_trg_sentences_tokenized,
                                           pretrain_train_ds.get_src_vocab().get_pad_idx(),
                                           pretrain_train_ds.get_trg_vocab().get_pad_idx(),
                                           pretrain_train_ds.get_trg_vocab().get_unk_idx())

    train_pretrain_data_loader = DataLoader(pretrain_train_ds, batch_size=train_args.pretrain_batch_size,
                                            num_workers=num_workers, shuffle=False,
                                            collate_fn=train_collator, sampler=train_sampler)

    val_sampler = DistributedSampler(dataset=pretrain_val_ds, seed=other_args.seed, drop_last=True,
                                     rank=rank, num_replicas=world_size)
    val_collator = Pretrain_USKI_Collator(pretrain_val_ds, pretrain_train_ds.get_src_vocab().get_pad_idx(),
                                          pretrain_train_ds.get_trg_vocab().get_pad_idx(),
                                          pretrain_train_ds.get_trg_vocab().get_unk_idx())
    val_pretrain_data_loader = DataLoader(pretrain_val_ds, batch_size=train_args.pretrain_batch_size,
                                          num_workers=num_workers,
                                          shuffle=False, collate_fn=val_collator, sampler=val_sampler)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1e-3)

    # # # # # # # #
    #       IDF computation
    # # # # # # # #
    document_frequencies = [0] * len(pretrain_train_ds.get_trg_vocab())
    train_trg_sentences = train_trg_sentences_tokenized
    for i in range(len(train_trg_sentences)):
        trg_sentence = train_trg_sentences[i].tolist()
        for token_idx in set(trg_sentence):   # remove duplicates do not increase more than once the count
            document_frequencies[token_idx] += 1

    inverse_document_frequencies = []
    for i in range(len(document_frequencies)):
        if document_frequencies[i] == 0:
            inverse_document_frequencies.append(0.0)  # se c'e' una parola non contata, allora era
            # del source e non ci interessa, metto a zero
        else:
            idf = np.log(len(train_trg_sentences)*1.2 / document_frequencies[i])  # document_frequencies[i])
            assert (idf >= 0), "should never be negative... lencorpus: " + str(len(train_trg_sentences)) + \
                               " document_frequencies[i]: " + str(document_frequencies[i]) + " idf: " + str(idf)
            inverse_document_frequencies.append(idf)

    tensor_idf = torch.tensor(inverse_document_frequencies).float()
    tensor_idf = tensor_idf / tensor_idf.sum()

    # small offset to avoid zero
    eps = 1e-4
    tensor_idf += eps

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1,
                                        ignore_index=pretrain_train_ds.get_trg_vocab().get_pad_idx(),
                                        weight=tensor_idf.to(rank))


    max_num_epoch = 100
    early_stop_iter = train_args.num_pretrain_iter
    global_iter = 0
    train_loss = 0
    start_time = time()
    for epoch in range(max_num_epoch):

        if global_iter > early_stop_iter:
            print("Reached early step iter, exiting pre-training")
            break

        for batch, it in zip(iter(train_pretrain_data_loader), range(len(train_pretrain_data_loader))):
            ddp_model.train()

            batch_src, num_pads_src, batch_trg, num_pads_trg, gt_answer = batch
            batch_input_x = batch_src.to(rank)
            batch_trg_y = batch_trg.to(rank)
            gt_answer = gt_answer.to(rank)

            _, pred = ddp_model(enc_input=batch_input_x, dec_input=batch_trg_y[:, :-1],
                                enc_input_num_pads=num_pads_src, dec_input_num_pads=num_pads_trg,
                                apply_softmax=False, mode='pretrain')

            loss = loss_function(pred.transpose(-1, -2), gt_answer[:, :-1])

            loss.backward()
            train_loss += loss.item()

            if (it+1) % train_args.pretrain_num_accum == 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_iter += 1
            if global_iter > early_stop_iter:
                print("Reached early step iter, exiting pre-training")
                break

            #
            if rank == 0 and ((global_iter) % save_args.save_pretrain_every_iter == 0):
                pretrain_save_file = save_args.save_path + "pretrained_model.pt"
                torch.save(ddp_model.state_dict(), pretrain_save_file)
                print("Saved checkpoint: " + str(pretrain_save_file))

            if (it+1) % train_args.print_every_iter == 0:
                print("Train epoch: " + str(epoch) + " - it " + str(global_iter) + " / " +
                      str(len(train_pretrain_data_loader)) +
                      " final loss: " + str(round(train_loss / global_iter, 6)), end=" ")

                val_loss = 0
                for batch in iter(val_pretrain_data_loader):
                    batch_src, num_pads_src, batch_trg, num_pads_trg, gt_answer = batch
                    batch_input_x = batch_src.to(rank)
                    batch_trg_y = batch_trg.to(rank)
                    gt_answer = gt_answer.to(rank)

                    _, pred = ddp_model(enc_input=batch_input_x, dec_input=batch_trg_y[:, :-1],
                                        enc_input_num_pads=num_pads_src, dec_input_num_pads=num_pads_trg,
                                        apply_softmax=False, mode='pretrain')

                    loss = loss_function(pred.transpose(-1, -2), gt_answer[:, :-1])

                    val_loss += loss.item()
                print("pre-train val: " + str(round(val_loss / len(val_pretrain_data_loader), 6)))




def nmt_train(rank, world_size,
              nmt_train_ds, nmt_val_ds, nmt_test_ds,
              ddp_model, train_args, sched_args, save_args, other_args):
    num_workers = 0

    train_sampler = DistributedSampler(dataset=nmt_train_ds, seed=other_args.seed, drop_last=True,
                                       rank=rank, num_replicas=world_size)
    train_collator = NmtBucketBatchCollator(nmt_train_ds.get_src_vocab().get_pad_idx(), nmt_train_ds.get_trg_vocab().get_pad_idx())
    # batch size is 1 because it is alreday organized in batches internally
    train_dl = DataLoader(nmt_train_ds, batch_size=1, num_workers=num_workers,
                          shuffle=False, collate_fn=train_collator, sampler=train_sampler)

    val_sampler = DistributedSampler(dataset=nmt_val_ds, seed=other_args.seed, drop_last=True,
                                     rank=rank, num_replicas=world_size)
    val_collator = NmtCollator(nmt_val_ds.get_src_vocab().get_pad_idx(), nmt_val_ds.get_trg_vocab().get_pad_idx())
    val_dl = DataLoader(nmt_val_ds, batch_size=train_args.eval_batch_size, num_workers=num_workers,
                        shuffle=False, collate_fn=val_collator, sampler=val_sampler)

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
            train_acc += acc / len(pred)
            train_loss += loss.item()
            sched_it += 1

            del batch_src, num_pads_src, batch_trg, num_pads_trg
            torch.cuda.empty_cache()
        train_loss = train_loss / len(train_dl)
        train_acc = train_acc / len(train_dl)
        print("Epoch: " + str(epoch) + " Train loss: " + str(round(train_loss, 4)) + " acc: " + str(round(train_acc, 4)), end=" | ")

        # evaluation
        if rank == 0 and ((epoch+1) % 10 == 0):

            # validation
            validation_loss = 0
            validation_acc = 0
            with torch.no_grad():
                ddp_model.eval()
                for batch, i in zip(iter(val_dl), range(len(val_dl))):
                    batch_src, num_pads_src, batch_trg, num_pads_trg = batch
                    batch_input_x = batch_src.to(rank)
                    batch_trg_y = batch_trg.to(rank)
                    pred = ddp_model(enc_input=batch_input_x, dec_input=batch_trg_y[:, :-1],
                                     enc_input_num_pads=num_pads_src, dec_input_num_pads=num_pads_trg,
                                     apply_softmax=False)
                    loss = loss_function(pred, batch_trg_y[:, 1:])
                    acc = compute_mean_accuracy(pred, batch_trg_y[:, 1:], num_pads_trg)
                    validation_loss += loss.item()
                    validation_acc += acc / len(pred)
                    del batch_src, num_pads_src, batch_trg, num_pads_trg
                    torch.cuda.empty_cache()

            validation_loss = validation_loss / len(val_dl)
            validation_acc = validation_acc / len(val_dl)
            print("Val loss: " + str(round(validation_loss, 4)) + " acc: " + str(round(validation_acc, 4)), end=" | ")

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

                      train_src_sentences_tokenized, train_trg_sentences_tokenized,

                      pretrain_train_ds, pretrain_val_ds,
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

    pretrain_save_file = save_args.save_path + "pretrained_model.pt"
    update_pretraining_file = False
    if not os.path.isfile(pretrain_save_file):
        print("File not found --- FIRST Pretraining")
        print("Pre-Training...")
        pretrain_USKI(rank, world_size,

                       train_src_sentences_tokenized, train_trg_sentences_tokenized,

                       nmt_val_ds,

                       pretrain_train_ds, pretrain_val_ds,
                       ddp_model, train_args, sched_args, save_args, other_args)
        torch.save(ddp_model.state_dict(), pretrain_save_file)
        print("First pretraing save -- Done.")
    else:
        ddp_model.load_state_dict(torch.load(pretrain_save_file), strict=False)
        print("Loaded ...Pre-Training...")
        if update_pretraining_file:
            pretrain_USKI(rank, world_size,

                           train_src_sentences_tokenized, train_trg_sentences_tokenized,

                           nmt_val_ds,

                           pretrain_train_ds, pretrain_val_ds,
                           ddp_model, train_args, sched_args, save_args, other_args)
            torch.save(ddp_model.state_dict(), pretrain_save_file)
            print("Pretrained model state updated.")

    pretrained_state_dict = model.state_dict()

    print("Classification model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
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

    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    print("Mapped pre-trained weights to Classification model")
    ddp_model.module.load_state_dict(pretrained_state_dict)

    print("Start NMT Training post Pre-Training...")
    nmt_train(rank, world_size, nmt_train_ds, nmt_val_ds, nmt_test_ds,
              ddp_model, train_args, sched_args, save_args, other_args)

    dist.destroy_process_group()


def train(nmt_train_ds, nmt_val_ds, nmt_test_ds,

          train_src_sentences_tokenized, train_trg_sentences_tokenized,

          pretrain_train_ds, pretrain_val_ds,
          model_args, train_args, sched_args,
          save_args, other_args):

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert (other_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(other_args.num_gpus))

    mp.spawn(distributed_train,
             args=(world_size,
                   nmt_train_ds, nmt_val_ds, nmt_test_ds,

                   train_src_sentences_tokenized, train_trg_sentences_tokenized,

                   pretrain_train_ds, pretrain_val_ds,
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

    parser.add_argument('--sched_type', type=scheduler_type_choice, default='noam')
    parser.add_argument('--warmup_steps', type=int, default=400)

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
    parser.add_argument('--save_pretrain_every_iter', type=int, default=20000)

    parser.add_argument('--parent_path', type=str, default='')
    parser.add_argument('--how_many_checkpoints', type=int, default=1)

    parser.add_argument('--max_seq_len', type=int, default=100)
    parser.add_argument('--language', type=str, default='kk-en')

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
                           trg_lang=data_args.trg_lang)
    sched_args = Namespace(sched_type=args.sched_type,
                           warmup_steps=args.warmup_steps,
                           how_many_checkpoints=args.how_many_checkpoints)
    save_args = Namespace(save_path=args.save_path,
                          save_pretrain_every_iter=args.save_pretrain_every_iter)
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #           Filter based on IoU > 10% s
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    src_sentences = []
    trg_sentences = []
    src_sentences_tokenized = []
    trg_sentences_tokenized = []

    from data.nmt import read_sentences_from_file_txt

    read_src_sentences = read_sentences_from_file_txt(data_args.data_path + data_args.src_train_filename)
    read_trg_sentences = read_sentences_from_file_txt(data_args.data_path + data_args.trg_train_filename)
    assert (len(src_sentences) == len(trg_sentences)), "src and trg should be the same"

    for i in range(len(read_src_sentences)):
        splitted_src_sentence = str.lower(read_src_sentences[i]).split()
        splitted_trg_sentence = str.lower(read_trg_sentences[i]).split()
        if len(splitted_src_sentence) < args.min_len or len(splitted_src_sentence) > args.max_len or \
                len(splitted_trg_sentence) < args.min_len or len(splitted_trg_sentence) > args.max_len:
            continue
        src_sentences.append(splitted_src_sentence)
        trg_sentences.append(splitted_trg_sentence)
        alarming_difference_ratio = 0.2
        if abs(len(src_sentences) - len(trg_sentences)) > \
                len(src_sentences) * alarming_difference_ratio:
            print("src and trg should roughly the same, otherwise it may be disaligned")
            print("src: " + str(src_sentences[i]))
            print("trg: " + str(trg_sentences[i]))
            print("Detected alarming difference...")
            exit(-1)

    for i in range(len(trg_sentences)):
        tmp = [trg_vocab.get_sos_str()] + trg_sentences[i] + [trg_vocab.get_eos_str()]
        tmp = trg_vocab.convert_word2idx_with_unk(tmp)
        trg_sentences_tokenized.append(torch.tensor(tmp))

    for i in range(len(src_sentences)):
        tmp = [src_vocab.get_sos_str()] + src_sentences[i] + [src_vocab.get_eos_str()]
        tmp = src_vocab.convert_word2idx_with_unk(tmp)
        src_sentences_tokenized.append(torch.tensor(tmp))

    import pickle
    with open('./iou_indexes_save/' + (args.language if not args.reverse_src_trg else args.language[::-1]) +  '_iou.pickle',
              'rb') as f:
        iou_idx_collection = pickle.load(f)

    if args.ALL_INDEXES:
        legal_indexes = []
        print("GENERATING ALL INDEXES:")
        for i in range(len(src_sentences)):
            for j in range(len(trg_sentences)):
                if i > j:
                    continue
                legal_indexes.append([i, j])
            if (i+1) % (len(src_sentences) // 10) == 0:
                print(str(i+1) + ' / ' + str(len(src_sentences)))
    else:
        legal_indexes = []
        for portion in range(1, 10):
            legal_indexes += iou_idx_collection[portion]  # discard just the first portion...

    pretrain_train_ds = PretrainSelectiveDataset(legal_indexes=legal_indexes,
                                                 src_vocab=src_vocab, trg_vocab=trg_vocab)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pretrain_val_ds = Pretrain_USKI_Dataset(src_sentences_path=data_args.data_path + data_args.src_val_filename,
                                            trg_sentences_path=data_args.data_path + data_args.trg_val_filename,
                                            min_length=args.min_len, max_length=args.max_len,
                                            src_vocab=src_vocab, trg_vocab=trg_vocab, verbose=True)

    train(nmt_train_ds, nmt_val_ds, nmt_test_ds,

          src_sentences_tokenized, trg_sentences_tokenized,

          pretrain_train_ds, pretrain_val_ds,
          model_args, train_args, sched_args, save_args, other_args)
