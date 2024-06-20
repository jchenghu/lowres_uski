
import torch
from utils.file_utils import read_sentences_from_file_txt

from torch.utils.data import Dataset


class NmtDataSet(Dataset):
    def __init__(self, src_sentences_path, trg_sentences_path, src_vocab, trg_vocab,
                 tokenize_trg=True,  # argument for evaluation
                 min_length=2, max_length=150,
                 apply_truncation=False,
                 verbose=False):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.src_sentences = []
        self.trg_sentences = []

        # let's simplify a little for the sake of debugging ----------------------------------
        read_src_sentences = read_sentences_from_file_txt(src_sentences_path)
        read_trg_sentences = read_sentences_from_file_txt(trg_sentences_path)
        assert (len(self.src_sentences) == len(self.trg_sentences)), "src and trg should be the same"
        # 16k, e 8k se fai lowering diventa uguale.

        for i in range(len(read_src_sentences)):
            splitted_src_sentence = str.lower(read_src_sentences[i]).split()
            splitted_trg_sentence = str.lower(read_trg_sentences[i]).split()
            if len(splitted_src_sentence) < min_length or len(splitted_src_sentence) > max_length or \
                    len(splitted_trg_sentence) < min_length or len(splitted_trg_sentence) > max_length:
                continue
            self.src_sentences.append(splitted_src_sentence)
            self.trg_sentences.append(splitted_trg_sentence)
            # OCCHIO che sto filtro non funziona, perche' col BPE e' capitato che ci sia un divario
            # enorme... quindi per qualche motivo sta cosa non sta filtrando..
            alarming_difference_ratio = 0.2
            if abs(len(self.src_sentences) - len(self.trg_sentences)) > \
                   len(self.src_sentences) * alarming_difference_ratio:
                print("src and trg should roughly the same, otherwise it may be disaligned")
                print("src: " + str(self.src_sentences[i]))
                print("trg: " + str(self.trg_sentences[i]))
                print("Detected alarming difference...")
                exit(-1)

        if verbose:
            print("------------------- Train Set before preproc -------------------------")
            for i in range(2):
                print(str(i) + ") -------------------------------------------- ")
                print("Src : " + str(self.src_sentences[i]))
                print("Trg : " + str(self.trg_sentences[i]))

        if tokenize_trg:
            for i in range(len(self.trg_sentences)):
                self.trg_sentences[i] = [self.trg_vocab.get_sos_str()] + self.trg_sentences[i] + [self.trg_vocab.get_eos_str()]
                self.trg_sentences[i] = self.trg_vocab.convert_word2idx_with_unk(self.trg_sentences[i])
                self.trg_sentences[i] = torch.tensor(self.trg_sentences[i])

        for i in range(len(self.src_sentences)):
            self.src_sentences[i] = [self.src_vocab.get_sos_str()] + self.src_sentences[i] + [self.src_vocab.get_eos_str()]
            self.src_sentences[i] = self.src_vocab.convert_word2idx_with_unk(self.src_sentences[i])
            self.src_sentences[i] = torch.tensor(self.src_sentences[i])

        if verbose:
            print("------------------- Train Set before preproc -------------------------")
            for i in range(2):
                print(str(i) + ") -------------------------------------------- ")
                print("Src : " + str(self.src_sentences[i]))
                print("Trg : " + str(self.trg_sentences[i]))

        if apply_truncation:
            print("Applying truncation...")
            new_src_sentences = []
            new_trg_sentences = []
            for i in range(len(self.src_sentences)):
                if len(self.src_sentences[i]) >= max_length or len(self.trg_sentences[i]) >= max_length:
                    continue
                else:
                    new_src_sentences.append(self.src_sentences[i])
                    new_trg_sentences.append(self.trg_sentences[i])
            self.src_sentences = new_src_sentences
            self.trg_sentences = new_trg_sentences

        self.max_len = 0
        for i in range(len(self.src_sentences)):
            self.max_len = max(self.max_len, len(self.src_sentences[i]))
        for i in range(len(self.trg_sentences)):
            self.max_len = max(self.max_len, len(self.trg_sentences[i]))
        if verbose:
            print("Len: " + str(len(self.src_sentences)))

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        return self.src_sentences[idx], self.trg_sentences[idx]

    def get_src_word2idx_dict(self):
        return self.src_vocab.get_word2idx_dict()

    def get_src_idx2word_list(self):
        return self.src_vocab.get_idx2word_list()

    def get_trg_word2idx_dict(self):
        return self.trg_vocab.get_word2idx_dict()

    def get_trg_idx2word_list(self):
        return self.trg_vocab.get_idx2word_list()

    def get_max_seq_len(self):
        return self.max_len

    def get_src_vocab(self):
        return self.src_vocab

    def get_trg_vocab(self):
        return self.trg_vocab


class NmtCollator:

    def __init__(self, src_pad_idx, trg_pad_idx, for_evaluation=False):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.for_evaluation = for_evaluation

    def __call__(self, batch):
        batch_src = []
        batch_trg = []
        src_length_list = []
        trg_length_list = []
        for item in batch:
            src_tensor, trg_tensor = item
            batch_src.append(src_tensor)
            batch_trg.append(trg_tensor)
            src_length_list.append(len(src_tensor))
            trg_length_list.append(len(trg_tensor))
        batch_src = torch.nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=self.src_pad_idx).long()
        if not self.for_evaluation:
            # during evaluation trg batch is actually the ground-truth string
            batch_trg = torch.nn.utils.rnn.pad_sequence(batch_trg, batch_first=True, padding_value=self.trg_pad_idx).long()
        num_pads_src = [max(src_length_list) - length for length in src_length_list]
        num_pads_trg = [max(trg_length_list) - length for length in trg_length_list]
        return batch_src, num_pads_src, batch_trg, num_pads_trg


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


import pickle

class NmtBucketBatchDataSet(Dataset):

    def __init__(self, src_sentences_path, trg_sentences_path, src_vocab, trg_vocab,
                 bucket_size, min_length=2, max_length=150,
                 verbose=False):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bucket_size = bucket_size

        # let's simplify a little for the sake of debugging ----------------------------------
        read_src_sentences = read_sentences_from_file_txt(src_sentences_path)
        read_trg_sentences = read_sentences_from_file_txt(trg_sentences_path)
        assert (len(read_src_sentences) == len(read_trg_sentences)), "src and trg should be the same"
        # 16k, e 8k se fai lowering diventa uguale.

        src_sentences = []
        trg_sentences = []
        for i in range(len(read_src_sentences)):
            splitted_src_sentence = str.lower(read_src_sentences[i]).split()
            splitted_trg_sentence = str.lower(read_trg_sentences[i]).split()
            if len(splitted_src_sentence) < min_length or len(splitted_src_sentence) > max_length or \
                    len(splitted_trg_sentence) < min_length or len(splitted_trg_sentence) > max_length:
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

        if verbose:
            print("------------------- Train Set before preproc -------------------------")
            for i in range(2):
                print(str(i) + ") -------------------------------------------- ")
                print("Src : " + str(src_sentences[i]))
                print("Trg : " + str(trg_sentences[i]))

        for i in range(len(trg_sentences)):
            trg_sentences[i] = [self.trg_vocab.get_sos_str()] + trg_sentences[i] + [self.trg_vocab.get_eos_str()]
            trg_sentences[i] = trg_vocab.convert_word2idx_with_unk(trg_sentences[i])
            #trg_sentences[i] = torch.tensor(trg_sentences[i])

        for i in range(len(src_sentences)):
            src_sentences[i] = [self.src_vocab.get_sos_str()] + src_sentences[i] + [self.src_vocab.get_eos_str()]
            src_sentences[i] = self.src_vocab.convert_word2idx_with_unk(src_sentences[i])
            #src_sentences[i] = torch.tensor(src_sentences[i])

        if verbose:
            print("------------------- Train Set before preproc -------------------------")
            for i in range(2):
                print(str(i) + ") -------------------------------------------- ")
                print("Src : " + str(src_sentences[i]))
                print("Trg : " + str(trg_sentences[i]))

        self.max_len = 0
        for i in range(len(src_sentences)):
            self.max_len = max(self.max_len, len(src_sentences[i]))
        for i in range(len(trg_sentences)):
            self.max_len = max(self.max_len, len(trg_sentences[i]))


        if verbose:
            print("Len: " + str(len(src_sentences)))

        # # # #
        # Perform BUCKETING
        sorted_indexes = sorted(list(range(len(trg_sentences))), key=lambda idx: (len(trg_sentences[idx])))
        self.src_batches = []
        self.trg_batches = []
        new_batch_src = []
        new_batch_trg = []
        for i in range(len(src_sentences)):
            # the number of tokens must consider also the paddings

            if len(new_batch_src) != 0:
                num_tokens_in_batch_src = max([len(sentence) for sentence in new_batch_src]) * len(new_batch_src)
                num_tokens_in_batch_trg = max([len(sentence) for sentence in new_batch_trg]) * len(new_batch_trg)
            else:
                num_tokens_in_batch_src = 0
                num_tokens_in_batch_trg = 0

            new_batch_src.append(src_sentences[sorted_indexes[i]])
            new_batch_trg.append(trg_sentences[sorted_indexes[i]])
            num_tokens_in_batch_src += len(src_sentences[sorted_indexes[i]])
            num_tokens_in_batch_trg += len(trg_sentences[sorted_indexes[i]])
            # the divisibility condition ensures all bpe_data loader on each proc are synchronized
            # over the same epoch, same iteration and share number of batches in one epoch.
            if num_tokens_in_batch_trg >= self.bucket_size:
                self.src_batches.append(new_batch_src)
                self.trg_batches.append(new_batch_trg)
                new_batch_src = []
                new_batch_trg = []

    def __len__(self):
        return len(self.src_batches)

    def __getitem__(self, idx):
        #     ForkingPickler(file, protocol).dump(obj)
        #   File "/home/jchu/.local/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 355, in reduce_storage
        #     metadata = storage._share_filename_cpu_()
        # RuntimeError: unable to mmap 608 bytes from file </torch_207329_3205365369_64763>: Cannot allocate memory (12)
        """
         Hello,
            I am just writing the workable solution for my case. I didnt do a deep analysis. From my dataloader, I was returning images(almost 12K images) and annotations (a list of size ~12K, each element in the list is a python dictionary) for every mini-batch. For 'file_system' sharing strategy, the workers dumped the fetched data in /dev/shm (mentioned above by @zejun-chen) and main process reads them via mmap. The error is coming from this reading. Instead of returning, a list for annotations, I wrapped it in the following way:
            return images, pickle.dumps(annotations). Then while reading, I just used (pickle.loads). Error is gone in this way.

            I get this idea from: this blog and this issue
                                            |           |
                                            |           https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
                                            |
                    https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        """
        return pickle.dumps([self.src_batches[idx], self.trg_batches[idx]])


    def get_src_word2idx_dict(self):
        return self.src_vocab.get_word2idx_dict()

    def get_src_idx2word_list(self):
        return self.src_vocab.get_idx2word_list()

    def get_trg_word2idx_dict(self):
        return self.trg_vocab.get_word2idx_dict()

    def get_trg_idx2word_list(self):
        return self.trg_vocab.get_idx2word_list()

    def get_max_seq_len(self):
        return self.max_len

    def get_src_vocab(self):
        return self.src_vocab

    def get_trg_vocab(self):
        return self.trg_vocab



class NmtBucketBatchCollator:

    def __init__(self, src_pad_idx, trg_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, batch):
        # batch_src, batch_trg = batch[0]
        loaded_elem = pickle.loads(batch[0])
        batch_src, batch_trg = loaded_elem

        src_length_list = []
        trg_length_list = []
        src_tensor_list = []
        trg_tensor_list = []
        for i in range(len(batch_src)):
            src_length_list.append(len(batch_src[i]))
            trg_length_list.append(len(batch_trg[i]))
            src_tensor_list.append(torch.tensor(batch_src[i]))
            trg_tensor_list.append(torch.tensor(batch_trg[i]))
        batch_src = torch.nn.utils.rnn.pad_sequence(src_tensor_list, batch_first=True, padding_value=self.src_pad_idx).long()
        batch_trg = torch.nn.utils.rnn.pad_sequence(trg_tensor_list, batch_first=True, padding_value=self.trg_pad_idx).long()
        num_pads_src = [max(src_length_list) - length for length in src_length_list]
        num_pads_trg = [max(trg_length_list) - length for length in trg_length_list]
        return batch_src, num_pads_src, batch_trg, num_pads_trg
