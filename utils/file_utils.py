
def read_sentences_from_file_txt(file_path, how_many=None, verbose=True):
    sentences = []
    with open(file_path, 'r') as f:
        counter = 0
        while True:
            line = f.readline()
            if line == '':
                break
            sentences.append(line[:-1])
            counter += 1
            if how_many is not None and counter == how_many:
                break
    if verbose:
        print(str(file_path) + " read " + str(len(sentences)) + " sentences")
    return sentences
