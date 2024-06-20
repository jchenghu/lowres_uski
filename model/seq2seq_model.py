
import torch
import torch.nn as nn


class Seq2Seq_Model(nn.Module):
    def __init__(self):
        super(Seq2Seq_Model, self).__init__()
        self.rank = None  # the device where the model is stored on

    def check_required_attributes(self):
        if self.rank is None:
            raise NotImplementedError("Subclass must assign the rank integer according to the GPU group")

    def forward_enc(self, enc_input, enc_input_num_pads,
                    is_pretraining=False):
        raise NotImplementedError

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=False,
                    is_pretraining=False):
        raise NotImplementedError

    # enc_x shape [batch_size, max_seq_len]
    # target_sentence [batch_size, max_seq_len]
    def forward(self, enc_input, dec_input=None,
                enc_input_num_pads=[0], dec_input_num_pads=[0], apply_log_softmax=False,
                mode='forward', **kwargs):
        if mode == 'forward':
            x = self.forward_enc(enc_input, enc_input_num_pads)
            y = self.forward_dec(x, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax)
            return y
        elif mode == 'pretrain':
            x = self.forward_enc(enc_input, enc_input_num_pads, is_pretraining=True)
            y = self.forward_dec(x, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax, is_pretraining=True)
            return x, y
        elif mode == 'beam_search':
            beta_arg = kwargs.get('beta', 5)
            how_many_outputs_arg = kwargs.get('how_many_outputs', 1)
            sample_or_max_arg = kwargs.get('sample_or_max', 'max')
            out_classes, out_logprobs = self.beam_search(
                enc_input, enc_input_num_pads,
                # gotta throw some exception in case the sos_idx is not initialized..
                sos_idx=kwargs.get('sos_idx', -1), eos_idx=kwargs.get('eos_idx', -1),
                beta=beta_arg,
                how_many_outputs=how_many_outputs_arg,
                sample_or_max=sample_or_max_arg)
            return out_classes, out_logprobs

    def beam_search(self, enc_input, enc_input_num_pads, sos_idx, eos_idx,
                    beta=3,
                    how_many_outputs=1,
                    sample_or_max='max'):
        assert (how_many_outputs <= beta), "requested output per sequence must be lower than beam width"
        assert (
                sample_or_max == 'max' or sample_or_max == 'sample'), "argument must be chosen between \'max\' and \'sample\'"
        bs = enc_input.shape[0]
        max_seq_len = min(enc_input.size(1) + 51, 190) # <--- LITTLE TRICK PER VOCAB SIZE PROBLEM. Max e' 200.
        #print("max_seq_len: " + str(max_seq_len) + " new max_seq_len: " + str(new_max_seq_len))
        #max_seq_len = new_max_seq_len

        # termination condition --------------------------------------------------------
        # There's no termination condition, just iter until max_seq_len. At every
        # iteration, keep just beta candidates for each sequence in the batch.

        # the cross_dec_input is used repeately, however when retrieving the output
        # we care only about one single decoded sententence for each input
        # therefore we can safely perform the encoder's forward right now
        cross_enc_output = self.forward_enc(enc_input, enc_input_num_pads)
        # [bs, enc_seq_len, d_model]

        # init: ------------------------------------------------------------------
        # [bs, 1]
        # print('SOS word: ' + str(y_word2idx_dict['SOS']))
        init_dec_class = torch.tensor([sos_idx] * bs).unsqueeze(1).type(torch.long).to(self.rank)
        init_dec_logprob = torch.tensor([0.0] * bs).unsqueeze(1).type(torch.float).to(self.rank)
        # log_probs: [bs, 1, num_class]
        log_probs = self.forward_dec(cross_input=cross_enc_output, enc_input_num_pads=enc_input_num_pads,
                                     dec_input=init_dec_class, dec_input_num_pads=[0] * bs,
                                     apply_log_softmax=True)
        if sample_or_max == 'max':
            _, topi = torch.topk(log_probs, k=beta, sorted=True)
        else:  # sample
            topi = torch.exp(log_probs[:, 0, :]).multinomial(num_samples=beta, replacement=False)
            topi = topi.unsqueeze(1)

        init_dec_class = init_dec_class.repeat(1, beta)  # [bs, beta]
        init_dec_class = init_dec_class.unsqueeze(-1)  # [bs, beta, 1]
        top_beta_class = topi.transpose(-2, -1)
        init_dec_class = torch.cat((init_dec_class, top_beta_class), dim=-1)  # [bs, beta, 2]

        init_dec_logprob = init_dec_logprob.repeat(1, beta)  # [bs, beta]
        init_dec_logprob = init_dec_logprob.unsqueeze(-1)  # [bs, beta, 1]
        top_beta_logprob = log_probs.gather(dim=-1, index=topi)  # [bs, 1, beta]
        top_beta_logprob = top_beta_logprob.transpose(-2, -1)  # [bs, beta, 1]
        init_dec_logprob = torch.cat((init_dec_logprob, top_beta_logprob), dim=-1)

        if type(cross_enc_output) is list:
            expanded_cross = []
            for cross in cross_enc_output:
                bs, enc_seq_len, d_model = cross.shape
                new_cross = cross.unsqueeze(1)  # [bs, 1, enc_seq_len, d_model]
                new_cross = new_cross.expand(-1, beta, -1, -1)  # [bs, beta, enc_seq_len, d_model]
                new_cross = new_cross.reshape(bs * beta, enc_seq_len, d_model).contiguous()  # -> [bs * beta, enc_seq_len, d_model]
                expanded_cross.append(new_cross)
            cross_enc_output = expanded_cross
        else:
            # expand the encoder output  vorrei diventi
            # [bs, enc_seq_len, d_model] -> [bs * beta, enc_seq_len, d_model]
            bs, enc_seq_len, d_model = cross_enc_output.shape
            cross_enc_output = cross_enc_output.unsqueeze(1)  # [bs, 1, enc_seq_len, d_model]
            cross_enc_output = cross_enc_output.expand(-1, beta, -1, -1)  # [bs, beta, enc_seq_len, d_model]
            cross_enc_output = cross_enc_output.reshape(bs * beta, enc_seq_len, d_model).contiguous()  # -> [bs * beta, enc_seq_len, d_model]

        # and modify also the pads accordingly
        enc_input_num_pads = [enc_input_num_pads[i] for i in range(bs) for _ in range(beta)]

        # loop: -----------------------------------------------------------------
        # "loop" prefix means that it changes during the loop, at time step 3, it will become [bs, beta, 3] and so on...
        loop_dec_classes = init_dec_class  # [bs, beta, 2]
        loop_dec_logprobs = init_dec_logprob  # [bs, beta, 2]
        loop_cumul_logprobs = loop_dec_logprobs.sum(dim=-1, keepdims=True)  # [bs, beta, 1]

        # this counts the number of VALID words in each sequence
        loop_num_elem_vector = torch.tensor([2] * (bs * beta)).to(self.rank)

        for time_step in range(2, max_seq_len):
            # at every time_step, loop_dec_input is of the form [bs, beta, time_step+1]
            # we need to reshape it into [bs*beta, time_step]
            loop_dec_classes = loop_dec_classes.reshape(bs * beta, time_step).contiguous()

            # [bs*beta, time_step, num_class]
            log_probs = self.forward_dec(cross_input=cross_enc_output, enc_input_num_pads=enc_input_num_pads,
                                         dec_input=loop_dec_classes,
                                         dec_input_num_pads=(time_step - loop_num_elem_vector).tolist(),
                                         apply_log_softmax=True)

            if sample_or_max == 'max':
                _, topi = torch.topk(log_probs[:, time_step - 1, :], k=beta, sorted=True)  # [bs * beta, 1, beta]
            else:  # sample
                topi = torch.exp(log_probs[:, time_step - 1, :]).multinomial(num_samples=beta,
                                                                             replacement=False)  # [bs * beta, beta_samples]

            # extract the top beta class given each sequence
            top_beta_word_classes = topi.reshape(bs, beta, beta)  # [bs, beta, beta]

            top_beta_word_logprobs = log_probs[:, time_step - 1, :].gather(dim=-1, index=topi)  # [bs*beta, 1, beta]
            top_beta_word_logprobs = top_beta_word_logprobs.reshape(bs, beta, beta)  # [bs, beta, beta]

            # each sequence have now its best prediction, but some sequence may have already been terminated with EOS,
            # in that case its prediction is simply ignored, and do not sum up in the "loop_dec_logprobs" their value
            # will be zero.
            # -------------------------------------- Apply Mask -------------------------------------
            # We apply mask to ignore the prediction of those candidates that are in an EOS state.
            # set to zero, the candidates given by the sequence that already have an EOS inside
            # [bs, bs, time_step] -> [bs, bs, time_step] booleans, it's filled with zeros, if a sequence have no EOS
            # in any of its time step. Then with the sum,
            there_is_eos_mask = (loop_dec_classes.view(bs, beta, time_step) == eos_idx). \
                sum(dim=-1, keepdims=True).type(torch.bool)  # [bs, beta, 1 ]
            # print(there_is_eos_mask.shape)
            # print(top_beta_word_logprobs.shape)
            # se c'e' un EOS nella sequenza, metto un mask

            # [bs * beta, 1 ] + [bs * beta, beta con le words azzerate se ho 'EOS']

            # mask the new top words coming from a sequence where there's already an EOS so they aren't going
            # to generate new candidates, since EOS represent the leaf in our exploration graph.
            # However, if we pad with -999 its candidates element, also the sequence containing EOS will be
            # straightforwardly discarded, we want to keep at least one, in the exploration, if we mask with 0.0 its
            # candidate would have maximum priority therefore we keep one word 0.0, and the other ones -999
            # so the
            top_beta_word_logprobs[:, :, 0:1].masked_fill_(there_is_eos_mask, 0.0)
            top_beta_word_logprobs[:, :, 1:].masked_fill_(there_is_eos_mask, -999.0)

            comparison_logprobs = loop_cumul_logprobs + top_beta_word_logprobs

            # [bs, beta] ogni beta deve confrontarsi con i nuovi beta, ottenendo cosi' una
            # [bs, beta, beta] dove [bs, i, j] ho la log prob della sequenza i + nuova classe j
            comparison_logprobs = comparison_logprobs.contiguous().view(bs,
                                                                        beta * beta)  # [bs, beta * beta] combination of each element

            _, topi = torch.topk(comparison_logprobs, k=beta, sorted=True)  # [bs, beta] indexes
            # extract which beta sequence the top sequence was constructed upon
            # and the top word he is referring to

            # /mnt/data/jchu_workspace/lowres_workspace/model/seq2seq_model.py:182: UserWarning: __floordiv__ is deprecated,
            # and its behavior will change in a future version of pytorch. It currently rounds toward 0
            # (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values.
            # To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division,
            # use torch.div(a, b, rounding_mode='floor').
            # which_sequence = topi // beta  # [bs, beta]
            #
            which_sequence = topi // beta  # [bs, beta]
            which_word = topi % beta  # [bs, beta]

            # we can now convert back classes and logprobs to the previous view
            loop_dec_classes = loop_dec_classes.view(bs, beta, -1)  # [bs, beta, time_step]
            loop_dec_logprobs = loop_dec_logprobs.view(bs, beta, -1)  # [bs, beta, time_step]

            # La sequenza j = 1...beta del batch size i=1...bs, deve essere sostituita con
            # La sequenza which_sequence[i,j] concatenato al nuovo vocabolo which_word[i,j]
            #   loop_dec_input shape: [bs, beta, time_step]
            #   new[i,j] = loop_dec_input[i,  which_sequence[i,j]]
            bs_idxes = torch.arange(bs).unsqueeze(-1)
            new_loop_dec_classes = loop_dec_classes[[bs_idxes, which_sequence]]  # [bs, beta, time_step]
            new_loop_dec_logprobs = loop_dec_logprobs[[bs_idxes, which_sequence]]  # [bs, beta, time_step]

            which_sequence_top_beta_word_classes = top_beta_word_classes[
                [bs_idxes, which_sequence]]  # [bs, beta, beta]
            which_sequence_top_beta_word_logprobs = top_beta_word_logprobs[
                [bs_idxes, which_sequence]]  # [bs, beta, beta]
            # which_word: [bs, beta] che ha un indice per ogni bs e beta, e
            which_word = which_word.unsqueeze(-1)  # [bs, beta, 1]

            lastword_top_beta_classes = which_sequence_top_beta_word_classes.gather(dim=-1,
                                                                                    index=which_word)  # [bs, beta, 1]
            lastword_top_beta_logprobs = which_sequence_top_beta_word_logprobs.gather(dim=-1, index=which_word)

            # update loop cumul logprobs
            # loop_cumul_logprobs = loop_cumul_logprobs + lastword_top_beta_logprobs  # [bs, beta, 1]

            # print("which word: " + str(which_top_beta_class))

            # update
            new_loop_dec_classes = torch.cat((new_loop_dec_classes, lastword_top_beta_classes), dim=-1)
            new_loop_dec_logprobs = torch.cat((new_loop_dec_logprobs, lastword_top_beta_logprobs), dim=-1)
            loop_dec_classes = new_loop_dec_classes
            loop_dec_logprobs = new_loop_dec_logprobs

            loop_cumul_logprobs = loop_dec_logprobs.sum(dim=-1, keepdims=True)

            # -----------------------update loop_num_elem_vector: ----------------------------
            # increase length of sequences, unless the last word indexed by loop num elem vector is EOS, in this case
            # we shouldn't increase the number of valid elements for that sequence! since all the words after EOS
            # are rubbish

            # devi incrementare solo se nella sequenza, NON c'e' un EOS nelle ultime lettere aggiunte,
            # quindi prima di questa qua nuova...
            loop_num_elem_vector = loop_num_elem_vector.view(bs, beta)[[bs_idxes, which_sequence]].view(bs * beta)
            there_was_eos_mask = (loop_dec_classes[:, :, :-1].view(bs, beta, time_step) == eos_idx). \
                sum(dim=-1).type(torch.bool).view(bs * beta)
            loop_num_elem_vector = loop_num_elem_vector + (1 * (1 - there_was_eos_mask.type(torch.int)))

            # early termination condition every elements in loop_num_elem_vector is lower than time_step
            # then we can break the for loop
            if (loop_num_elem_vector != time_step + 1).sum() == (bs * beta):
                break


        # sort out the best result
        loop_cumul_logprobs /= loop_num_elem_vector.reshape(bs, beta, 1)
        _, topi = torch.topk(loop_cumul_logprobs.squeeze(-1), k=beta)
        finished_sentences_class = [[] for i in range(bs)]
        finished_sentences_logprob = [[] for i in range(bs)]
        for i in range(bs):
            for j in range(how_many_outputs):
                idx = topi[i, j].item()
                finished_sentences_class[i].append(
                    loop_dec_classes[i, idx, :loop_num_elem_vector[i * beta + idx]].tolist())
                finished_sentences_logprob[i].append(
                    loop_dec_logprobs[i, idx, :loop_num_elem_vector[i * beta + idx]])

        return finished_sentences_class, finished_sentences_logprob
