
import torch
from torch.utils.data import Dataset


class PretrainSelectiveDataset(Dataset):
    def __init__(self, legal_indexes, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.legal_indexes = legal_indexes
        self.num_samples = len(legal_indexes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.legal_indexes[idx]

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


class PretrainSelectiveCollator:
    def __init__(self,
                 src_sentences_tokenized, trg_sentences_tokenized,
                 src_pad_idx, trg_pad_idx, trg_unk_idx):
        """
            Passare le tokenized sentences qui dovrebbe evitarmi gli errori di shared memory
            del dataloader, num_workers etc...
        """
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sentences_tokenized = src_sentences_tokenized
        self.trg_sentences_tokenized = trg_sentences_tokenized
        self.trg_unk_idx = trg_unk_idx

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
            outer_idx, inner_idx = idx
            OUTER_src_tensor = self.src_sentences_tokenized[outer_idx]
            INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx]
            batch_src.append(OUTER_src_tensor)
            batch_trg.append(INNER_trg_tensor)
            src_length_list.append(len(OUTER_src_tensor))
            trg_length_list.append(len(INNER_trg_tensor))

            # create the Label
            # compute IoU as the ground-truth from the string version
            # OUTER_src_tensor = self.src_sentences[outer_idx]
            # INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx].tolist()
            OUTER_trg_tensor = self.trg_sentences_tokenized[outer_idx]
            # INNER_src_tensor = self.src_sentences[inner_idx]

            idx_intersection = self.compute_token_iou(OUTER_trg_tensor[1:-1].tolist(),
                                                      INNER_trg_tensor[1:-1].tolist())
            # padda gli elementi non inclusi...
            batch_label_y.append(torch.tensor([idx if idx != self.trg_unk_idx and (idx in idx_intersection) else self.trg_pad_idx
                                               for idx in INNER_trg_tensor.tolist()]))

        batch_src = torch.nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=self.src_pad_idx).long()
        batch_trg = torch.nn.utils.rnn.pad_sequence(batch_trg, batch_first=True, padding_value=self.trg_pad_idx).long()
        num_pads_src = [max(src_length_list) - length for length in src_length_list]
        num_pads_trg = [max(trg_length_list) - length for length in trg_length_list]
        batch_label_y = torch.nn.utils.rnn.pad_sequence(batch_label_y, batch_first=True, padding_value=self.trg_pad_idx).long()
        return batch_src, num_pads_src, batch_trg, num_pads_trg, batch_label_y


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------




import pickle

class BucketBatchPretrainSelectiveDataset(Dataset):


    def __init__(self, legal_indexes, src_vocab, trg_vocab,
                 src_sentences, trg_sentences,
                 bucket_size):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.legal_indexes = legal_indexes

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bucket_size = bucket_size

        max_src_len = []
        for i in range(len(legal_indexes)):
            OUTER_idx, _ = legal_indexes[i]
            max_src_len.append(len(src_sentences[OUTER_idx]))
        max_src_len = max(max_src_len)


        range_intervals = [arange_i * 0.1 * max_src_len for arange_i in list(range(10))]
        partition_group = [[] for _ in range(10)]
        for i in range(len(legal_indexes)):
            OUTER_idx, _ = legal_indexes[i]
            src_len = len(src_sentences[OUTER_idx])
            for q in range(len(range_intervals)):
                lower_end = range_intervals[q-1] if q != 0 else 0
                upper_end = range_intervals[q]
                if lower_end < src_len <= upper_end:
                    partition_group[q].append(i)

        print("How many elements are in each interval:")
        for q in range(len(partition_group)):
            print("group n°" + str(q) + ": " + str(len(partition_group[q])) + " elements")

        print("Sorting...")
        self.legal_indexes_batches = []
        for q in range(len(range_intervals)):

            # SORT INDEXES WITHIN RANGE INTERVALS...
            legal_index_ratio_len = []
            for legal_index_i in partition_group[q]:
                _, INNER_idx = legal_indexes[legal_index_i]
                legal_index_ratio_len.append(len(trg_sentences[INNER_idx]))

            sorted_indexes = sorted(range(len(partition_group[q])), key=lambda idx: legal_index_ratio_len[idx])

            new_batch_src_len = []
            new_batch_trg_len = []

            new_batch = []
            num_tokens_in_batch = 0
            for sorted_i in sorted_indexes:
                legal_index_i = partition_group[q][sorted_i]
                OUTER_idx, INNER_idx = legal_indexes[legal_index_i]
                # the number of tokens must consider also the paddings

                if num_tokens_in_batch <= self.bucket_size:
                    new_batch.append((OUTER_idx, INNER_idx))
                    new_batch_src_len.append(len(src_sentences[OUTER_idx]))
                    new_batch_trg_len.append(len(trg_sentences[INNER_idx]))
                    num_tokens_in_batch = (max(new_batch_src_len) + max(new_batch_trg_len)) * len(new_batch)
                else:
                    self.legal_indexes_batches.append(new_batch)
                    new_batch = []
                    new_batch_src_len = []
                    new_batch_trg_len = []
                    num_tokens_in_batch = 0

        self.num_samples = len(self.legal_indexes_batches)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return pickle.dumps(self.legal_indexes_batches[idx])

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


class BucketBatchPretrainSelectiveCollator:
    def __init__(self,
                 src_sentences_tokenized, trg_sentences_tokenized,
                 src_pad_idx, trg_pad_idx, trg_unk_idx):

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_sentences_tokenized = src_sentences_tokenized
        self.trg_sentences_tokenized = trg_sentences_tokenized
        self.trg_unk_idx = trg_unk_idx

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
        loaded_elem = pickle.loads(batch[0])
        for i in range(len(loaded_elem)):
            outer_idx, inner_idx = loaded_elem[i]
            # outer_idx, inner_idx = idx
            OUTER_src_tensor = self.src_sentences_tokenized[outer_idx]
            INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx]
            batch_src.append(OUTER_src_tensor)
            batch_trg.append(INNER_trg_tensor)
            src_length_list.append(len(OUTER_src_tensor))
            trg_length_list.append(len(INNER_trg_tensor))

            # create the Label
            # compute IoU as the ground-truth from the string version
            # OUTER_src_tensor = self.src_sentences[outer_idx]
            # INNER_trg_tensor = self.trg_sentences_tokenized[inner_idx].tolist()
            OUTER_trg_tensor = self.trg_sentences_tokenized[outer_idx]
            # INNER_src_tensor = self.src_sentences[inner_idx]

            # compute IoU between OUTER_trg_tensor and INNER_trg_tensor
            # [1:-1] removes SOS and EOS idx...
            idx_intersection = self.compute_token_iou(OUTER_trg_tensor[1:-1].tolist(),
                                                      INNER_trg_tensor[1:-1].tolist())

            batch_label_y.append(torch.tensor([idx if idx != self.trg_unk_idx and (idx in idx_intersection) else self.trg_pad_idx
                                               for idx in INNER_trg_tensor.tolist()]))

        batch_src = torch.nn.utils.rnn.pad_sequence(batch_src, batch_first=True, padding_value=self.src_pad_idx).long()
        batch_trg = torch.nn.utils.rnn.pad_sequence(batch_trg, batch_first=True, padding_value=self.trg_pad_idx).long()
        num_pads_src = [max(src_length_list) - length for length in src_length_list]
        num_pads_trg = [max(trg_length_list) - length for length in trg_length_list]
        batch_label_y = torch.nn.utils.rnn.pad_sequence(batch_label_y, batch_first=True, padding_value=self.trg_pad_idx).long()
        return batch_src, num_pads_src, batch_trg, num_pads_trg, batch_label_y

