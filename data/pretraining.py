
import torch
from torch.utils.data import Dataset


class Pretrain_USKI_Dataset(Dataset):
    def __init__(self, src_sentences_path, trg_sentences_path, src_vocab, trg_vocab,
                 min_length=2, max_length=150,
                 verbose=False):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # these aren't given in the __get__ function!
        self.src_sentences = []
        self.trg_sentences = []
        self.src_sentences_tokenized = []
        self.trg_sentences_tokenized = []

        # let's simplify a little for the sake of debugging ----------------------------------
        from data.nmt import read_sentences_from_file_txt
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
            alarming_difference_ratio = 0.2
            if abs(len(self.src_sentences) - len(self.trg_sentences)) > \
                   len(self.src_sentences) * alarming_difference_ratio:
                print("src and trg should roughly the same, otherwise it may be disaligned")
                print("src: " + str(self.src_sentences[i]))
                print("trg: " + str(self.trg_sentences[i]))
                print("Detected alarming difference...")
                exit(-1)

        for i in range(len(self.trg_sentences)):
            tmp = [self.trg_vocab.get_sos_str()] + self.trg_sentences[i] + [self.trg_vocab.get_eos_str()]
            tmp = self.trg_vocab.convert_word2idx_with_unk(tmp)
            self.trg_sentences_tokenized.append(torch.tensor(tmp))

        for i in range(len(self.src_sentences)):
            tmp = [self.src_vocab.get_sos_str()] + self.src_sentences[i] + [self.src_vocab.get_eos_str()]
            tmp = self.src_vocab.convert_word2idx_with_unk(tmp)
            self.src_sentences_tokenized.append(torch.tensor(tmp))

        self.num_samples = len(self.src_sentences)

    def __len__(self):
        return self.num_samples * self.num_samples

    def __getitem__(self, idx):
        return idx

    def get_src_word2idx_dict(self):
        return self.src_vocab.get_word2idx_dict()

    def get_src_idx2word_list(self):
        return self.src_vocab.get_idx2word_list()

    def get_trg_word2idx_dict(self):
        return self.trg_vocab.get_word2idx_dict()

    def get_trg_idx2word_list(self):
        return self.trg_vocab.get_idx2word_list()

    def get_src_vocab(self):
        return self.src_vocab

    def get_trg_vocab(self):
        return self.trg_vocab


class Pretrain_USKI_Collator:
    def __init__(self, pretrain_uski_dataset,
                src_pad_idx, trg_pad_idx, trg_unk_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sentences = pretrain_uski_dataset.src_sentences
        self.trg_sentences = pretrain_uski_dataset.trg_sentences
        self.src_sentences_tokenized = pretrain_uski_dataset.src_sentences_tokenized
        self.trg_sentences_tokenized = pretrain_uski_dataset.trg_sentences_tokenized
        self.trg_unk_idx = trg_unk_idx
        self.dataset_size = len(self.src_sentences)

    def compute_token_iou(self, sentence_A, sentence_B):
        unique_A_set = set(sentence_A)
        unique_B_set = set(sentence_B)
        intersection = unique_A_set & unique_B_set
        return intersection

    def __call__(self, batch):
        batch_src = []
        batch_trg = []
        src_length_list = []
        trg_length_list = []
        batch_label_y = []
        for idx in batch:
            outer_idx = idx // self.dataset_size
            inner_idx = idx % self.dataset_size

            OUTER_src_tensor = self.src_sentences_tokenized[outer_idx]
            INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx]
            batch_src.append(OUTER_src_tensor)
            batch_trg.append(INNER_trg_tensor)
            src_length_list.append(len(OUTER_src_tensor))
            trg_length_list.append(len(INNER_trg_tensor))

            # create the Label
            # compute IoU as the ground-truth from the string version
            # OUTER_src_tensor = self.src_sentences[outer_idx]
            INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx].tolist()
            OUTER_trg_tensor = self.trg_sentences_tokenized[outer_idx].tolist()
            # INNER_src_tensor = self.src_sentences[inner_idx]

            # compute IoU between OUTER_trg_tensor and INNER_trg_tensor
            # [1:-1] removes SOS and EOS idx...
            idx_intersection = self.compute_token_iou(OUTER_trg_tensor[1:-1], INNER_trg_tensor[1:-1])

            # padda gli elementi non inclusi...
            batch_label_y.append(torch.tensor([idx if idx != self.trg_unk_idx and (idx in idx_intersection) else self.trg_pad_idx for idx in INNER_trg_tensor]))

        batch_src = torch.nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=self.src_pad_idx).long()
        batch_trg = torch.nn.utils.rnn.pad_sequence(batch_trg, batch_first=True, padding_value=self.trg_pad_idx).long()
        num_pads_src = [max(src_length_list) - length for length in src_length_list]
        num_pads_trg = [max(trg_length_list) - length for length in trg_length_list]
        batch_label_y = torch.nn.utils.rnn.pad_sequence(batch_label_y, batch_first=True, padding_value=self.trg_pad_idx).long()
        return batch_src, num_pads_src, batch_trg, num_pads_trg, batch_label_y


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

