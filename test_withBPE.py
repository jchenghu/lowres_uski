
import sacrebleu
from sacremoses import MosesDetokenizer


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.nmt import NmtDataSet, NmtCollator
import math
from time import time

def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m "


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mean_accuracy(pred, target, dec_x_num_pads):
    batch_size, dec_len, vocab_size = pred.shape
    cumul_accuracy = 0
    probs = F.softmax(pred, dim=-1)
    _, topi = torch.topk(probs, k=1)
    topi = topi.squeeze(-1)  # [bs, dec_len]

    for i in range(batch_size):
        num_non_pad_token = dec_len - dec_x_num_pads[i]
        correct_pred = (topi[i, :num_non_pad_token] == target[i, :num_non_pad_token])
        if num_non_pad_token == 0:
            continue  # in somecases, batches contain empty string in both input and output sequence, thus
                      # an entire row filled with pads should not be considered in the accuracy computation
        else:
            accuracy = correct_pred.sum(dim=-1, keepdim=False).item() / num_non_pad_token
        if accuracy < 0:
            print("wtf acc: " + str(accuracy) + " correct_pred: " + str(correct_pred) + "topi: " + str(topi))
        cumul_accuracy += accuracy
    return cumul_accuracy


def evaluate_model_withBPE(model, nmt_ds, trg_lang, beta, verbose=True):
    if verbose:
        print("Evaluation Phase over " + str(len(nmt_ds)) + " sentences-----------------------")
    eval_timer_start = time()

    nmt_collator = NmtCollator(nmt_ds.get_src_vocab().get_pad_idx(), nmt_ds.get_trg_vocab().get_pad_idx(), for_evaluation=True)
    nmt_dl = DataLoader(nmt_ds, batch_size=32, num_workers=1, shuffle=False, collate_fn=nmt_collator)

    list_predictions = []
    list_list_references = []

    for batch in iter(nmt_dl):
        batch_src, num_pads_src, batch_trg, num_pads_trg = batch

        batch_src = torch.tensor(batch_src).to(model.rank)

        beam_search_kwargs = {'beta': beta,
                              'sample_or_max': 'max',
                              'sos_idx': nmt_ds.get_trg_vocab().get_pad_idx(),
                              'eos_idx': nmt_ds.get_trg_vocab().get_eos_idx(),
                              'how_many_outputs': 1}
        output_words, _ = model(enc_input=batch_src,
                                enc_input_num_pads=num_pads_src,
                                mode='beam_search', **beam_search_kwargs)
        output_words = [output_words[i][0] for i in range(len(output_words))]

        for i in range(len(batch_src)):
            # remove </w> from bpe notation preparing sentences for a clean output
            # hence 1:-1: we remove SOS and EOS
            pred = nmt_ds.get_trg_vocab().convert_idx2word(output_words[i][1:-1])
            target = batch_trg[i]

            #  E.g. ['SOS', 'In', 'school', ',', 'we', 'spent', 'a', 'lot', 'of', 'time',
            #  'stud@@', 'ying', 'the', 'history', 'of', 'K@@', 'im', 'I@@', 'l-@@', 'S@@',
            #  'ung', ',', 'but', 'we', 'never', 'learned', 'much', 'about', 'the',
            #  'outside', 'world', ',', 'ex@@', 'cept', 'that', 'America', ',', 'South',
            #  'K@@', 'o@@', 're@@', 'a', ',', 'J@@', 'ap@@', 'an', 'are', 'the', 'ene@@',
            #  'mi@@', 'es', '.', 'EOS']
            pred = (''.join(' '.join(pred).split('@@ '))).split()
            target = (''.join(' '.join(target).split('@@ '))).split()

            list_predictions.append(pred)
            list_list_references.append([target])

    for i in range(4):
        input_sentence, target_sentence = list(nmt_ds)[i]   # fetch some examples
        input_string = nmt_ds.get_src_vocab().convert_idx2word(input_sentence[1:-1])
        print(str(i) + ") -------------------------- " + "Input: " + str(input_string) + \
              " \nPred: " + str(list_predictions[i]) + " \nGt: " + str(list_list_references[i]), flush=True)

    # apply Moses Detokenizer to both prediction and references
    md = MosesDetokenizer(trg_lang)
    sacrebleu_list_list_references = [[md.detokenize([' '.join(ref[0])]) for ref in list_list_references]]
    sacrebleu_list_predictions = [md.detokenize([' '.join(pred)]) for pred in list_predictions]
    sacrebleu_score = sacrebleu.corpus_bleu(sacrebleu_list_predictions, sacrebleu_list_list_references)

    if verbose:
        print("SacreBLEU: " + str(sacrebleu_score))

    return str(sacrebleu_score).split()[2]


def eval_compute_score_on_set_withBPE(ddp_model, nmt_eval_ds,
                                      beam_size,
                                      trg_lang,
                                      verbose=False):

    with torch.no_grad():
        ddp_model.eval()
        sacrebleu_score = evaluate_model_withBPE(ddp_model, nmt_eval_ds, beta=beam_size, trg_lang=trg_lang, verbose=verbose)
        return sacrebleu_score

