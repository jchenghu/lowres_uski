
from utils.file_utils import read_sentences_from_file_txt


class VocabTokenizer:

    EOS_str = '<EOS>'
    PAD_str = '<PAD>'
    UNK_str = '<UNK>'
    SOS_str = '<SOS>'

    def __init__(self,
                 sentences_path,
                 verbose=True):

        # let's simplify a little for the sake of debugging ----------------------------------
        sentences = read_sentences_from_file_txt(sentences_path)

        for i in range(len(sentences)):
            sentences[i] = sentences[i].split()

        self.word2idx_dict = {VocabTokenizer.PAD_str: 0, VocabTokenizer.SOS_str: 1,
                              VocabTokenizer.EOS_str: 2, VocabTokenizer.UNK_str: 3}
        self.idx2word_list = [VocabTokenizer.PAD_str, VocabTokenizer.SOS_str,
                              VocabTokenizer.EOS_str, VocabTokenizer.UNK_str]
        for i in range(len(sentences)):
            for word in sentences[i]:
                if word not in self.word2idx_dict.keys():
                    self.word2idx_dict[word] = len(self.word2idx_dict)
                    self.idx2word_list.append(word)

        if verbose:
            print("Dictionary generated. Shared_word2idx_dict len: " + str(len(self.word2idx_dict)))
            print("Word examples: [" + self.idx2word_list[0] + ", " + self.idx2word_list[1] + ", " + self.idx2word_list[2] + "]")

    def __len__(self):
        return len(self.idx2word_list)

    def get_word2idx_dict(self):
        return self.word2idx_dict

    def get_idx2word_list(self):
        return self.idx2word_list

    def convert_word2idx_with_unk(self, sentence):
        new_sentence = []
        for word in sentence:
            if word in self.word2idx_dict.keys():
                new_sentence.append(self.word2idx_dict[word])
            else:
                new_sentence.append(self.get_unk_idx())
        return new_sentence

    def convert_idx2word(self, sentence):
        new_sentence = []
        for word in sentence:
            new_sentence.append(self.idx2word_list[word])
        return new_sentence

    def get_unk_str(self):
        return VocabTokenizer.UNK_str

    def get_pad_str(self):
        return VocabTokenizer.PAD_str

    def get_sos_str(self):
        return VocabTokenizer.SOS_str

    def get_eos_str(self):
        return VocabTokenizer.EOS_str

    def get_unk_idx(self):
        return self.word2idx_dict[self.get_unk_str()]

    def get_pad_idx(self):
        return self.word2idx_dict[self.get_pad_str()]

    def get_sos_idx(self):
        return self.word2idx_dict[self.get_sos_str()]

    def get_eos_idx(self):
        return self.word2idx_dict[self.get_eos_str()]


class SharedVocabTokenizer:
    EOS_str = '<EOS>'
    PAD_str = '<PAD>'
    UNK_str = '<UNK>'
    SOS_str = '<SOS>'

    def __init__(self,
                 sentences_path_src,
                 sentences_path_trg,
                 verbose=True):

        # let's simplify a little for the sake of debugging ----------------------------------
        sentences_src = read_sentences_from_file_txt(sentences_path_src)
        sentences_trg = read_sentences_from_file_txt(sentences_path_trg)

        for i in range(len(sentences_src)):
            sentences_src[i] = sentences_src[i].split()

        for i in range(len(sentences_trg)):
            sentences_trg[i] = sentences_trg[i].split()

        self.word2idx_dict = {VocabTokenizer.PAD_str: 0, VocabTokenizer.SOS_str: 1,
                              VocabTokenizer.EOS_str: 2, VocabTokenizer.UNK_str: 3}
        self.idx2word_list = [VocabTokenizer.PAD_str, VocabTokenizer.SOS_str,
                              VocabTokenizer.EOS_str, VocabTokenizer.UNK_str]
        for i in range(len(sentences_src)):
            for word in sentences_src[i]:
                if word not in self.word2idx_dict.keys():
                    self.word2idx_dict[word] = len(self.word2idx_dict)
                    self.idx2word_list.append(word)
        for i in range(len(sentences_trg)):
            for word in sentences_trg[i]:
                if word not in self.word2idx_dict.keys():
                    self.word2idx_dict[word] = len(self.word2idx_dict)
                    self.idx2word_list.append(word)

        if verbose:
            print("Dictionary generated. Shared_word2idx_dict len: " + str(len(self.word2idx_dict)))
            print("Word examples: [" + self.idx2word_list[0] + ", " + self.idx2word_list[1] + ", " + self.idx2word_list[
                2] + "]")

    def __len__(self):
        return len(self.idx2word_list)

    def get_word2idx_dict(self):
        return self.word2idx_dict

    def get_idx2word_list(self):
        return self.idx2word_list

    def convert_word2idx_with_unk(self, sentence):
        new_sentence = []
        for word in sentence:
            if word in self.word2idx_dict.keys():
                new_sentence.append(self.word2idx_dict[word])
            else:
                new_sentence.append(self.get_unk_idx())
        return new_sentence

    def convert_idx2word(self, sentence):
        new_sentence = []
        for word in sentence:
            new_sentence.append(self.idx2word_list[word])
        return new_sentence

    def get_unk_str(self):
        return VocabTokenizer.UNK_str

    def get_pad_str(self):
        return VocabTokenizer.PAD_str

    def get_sos_str(self):
        return VocabTokenizer.SOS_str

    def get_eos_str(self):
        return VocabTokenizer.EOS_str

    def get_unk_idx(self):
        return self.word2idx_dict[self.get_unk_str()]

    def get_pad_idx(self):
        return self.word2idx_dict[self.get_pad_str()]

    def get_sos_idx(self):
        return self.word2idx_dict[self.get_sos_str()]

    def get_eos_idx(self):
        return self.word2idx_dict[self.get_eos_str()]

