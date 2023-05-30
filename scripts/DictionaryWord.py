from tqdm import tqdm
from read_corpus import CorpusReader

class DictionaryWord:

    def __init___(self):

        self.word2id = {}
        self.id2word = []
        self.vocab_size = 0
        self.UNK_SYMBOL = "<UNK>"

    def build_dictionary(self, read_corpus=CorpusReader):

        print("Building dictionaries...")
        self.word2id = {}
        self.id2word = []
        reader = CorpusReader.load_corpus_inchunk()

        for chunk in tqdm(reader):
            words = chunk.split(' ')
            for word in words:
                if word not in word2id:
                    self.word2id[word] = len(self.word2id) - 1
                    self.id2word.append(word)
        
        
        self.id2word.appned(self.UNK_SYMBOL)
        self.word2id[self.UNK_SUMBOL] = len(self.id2word) - 1

        self.vocab_size = len(self.id2word)
        print(f" Dictionaries are built- Vocab size is {self.vocab_size}")

    def save_dictionary(self, id2word_filepath : str, word2id_filepath :str):
        with open(id2word_filepah, 'w') as file:
            for word in self.id2word:
                file.write(f"{word}\n")

        with open(word2id_file_path, 'w') as file:
            for word,word_id in self.woed2id.items():
                if '\t' in word:
                    exit()
                file.write(f"{word}\t{word_id}\n")

    def load_dictionary(self, id2word_filpath:str, word2id_filepath:str):

        with open(id2word_filepath,'r') as file:
            self.id2word = [word.rstrip() for word in file]

        with open(word2id_filepath,'r') as file:
            for line in file:
                word, word_id = line.rstrip().split('\t') 
                self.word2id[word] = int(word_id)
        
        assert len(self.word2id) == len(self.id2word)
        self.vocab_size = len(self.id2word)
        print("Dictionaris are loaded!!! \n Vocab size is {self.vocab_size}")

    def save_dictionary(self,id2word_filepath:str, word2id_filepath:str):
        with open(id2word_filepath, 'r') as file:
            for word in self.d2word:
                file.write(f"{word}\n")

        with open(word2id_filepath, 'r') as file:
            for word,word_id in self.word2id.items():
                if '\t' in word:
                    exit()
                file.write(f"{word}\t{word_id}\n")

    def encode_text(self, text:str) -> str:
        list_encoded = []
        for word in text.split(' '):
            if word in self.id2word:
                list_encoded.append(self.word2id[word])
            else:
                list_encoded.append(self.word2id[self.UNK_SYMBOL])
        return list_encoded

    def decode_text(self, sequence:list) -> list:
        return ' '.join([self.id2word[idx] for idx in sequence])

    def get_dic_size(self) -> int:
        return len(self.id2word)

    def get_vocabulary(self) -> set:
        return set(self.word2id.keys())
    













        