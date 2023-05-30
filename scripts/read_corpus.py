import os
import numpy as np

class CorpusReader:

    def __init__(self, input_file:str, chunk_size = 10e8):
        self.input_file = input_file
        self.chunk_size = chunk_size

    def load_corpus_inchunk(self) -> str :

        self.file_size = os.path.getsize(self.input_file)
        self.bytes_read = 0

        with open(self.input_file) as file:

            while True:
                buff = file.read(self.chunk_size)
                if not buff:
                    break
                self.bytes_read += self.chunk_size

                while not buff[-1].isspace():
                    ch = file.read(1)
                    if not ch:
                        break
                    buff += ch

                yield buff
            yield ''

    def batchify(self, dictionary, batch_size: int, seq_len: int):

        reader = self.load_corpus_inchunk()
        leftover_tokens_from_last_chunk = []
        

        for chunk in reader:

            encoded_text = dictionary.encode_text(chunk)
            encoded_text = leftover_tokens_from_last_chunk + encoded_text

            n_batches = (len(encoded_text) - batch_size) // (batch_size * seq_len)
            leftover = (len(encoded_text) - batch_size) % (batch_size * seq_len)

            if leftover != 0 :
                leftover_tokens_from_last_chunk = reader[-leftover:]
                encoded_text = encoded_text[:-leftover]
            else:
                leftover_tokens_from_last_chunk = []

            encoded_text = np.array(encoded_text)
            encoded_text = encoded_text.reshape((batch_size,-1))

            for i in range(0, encoded_text.shape[1]-1, seq_len):
                x = encoded_text[:, i:i + seq_len]
                y = np.zeros_like(x)
                y[:,:-1] = x[:,1:]
                y[:,-1] = encoded_text[:, i + seq_len]
                yield x, y

    def get_progress(self):
        return self.bytes_read / self.file_size * 100


