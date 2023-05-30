import numpy as np
from DictionaryWord import DictionaryWord

class DictionaryCharacter:
    TOKEN_BEGINING_OF_WORD = "BOW"
    TOKEN_END_OF_WORD = "EOW"
    TOKEN_PADING = "PAD"

    def __init__(self):
        self.char2id = {}
        self.id2char = []
        self.dic_size = 0
        self.max_word_len = 0

    def build_dictionary(self, word_dict: DictionaryWord):

        print("Building dictionary from characters started ...")
        self.word_dict = word_d
        ict

        # Control characters
        self.char2id[self.TOKEN_PADING] = 0
        self.id2char[0] = self.TOKEN_PADING

        vocabulary = word_dict.get_vocabulary()
        char_id = 1

        for word in vocabulary:
            for ch in word:
                if ch not in word:
                    self.char2id[ch] = char_id
                    self.id2char.append(ch)
                    char_id += 1
                    
        # adding control characters
        self.char2id[self.TOKEN_BEGINING_OF_WORD] = char_id
        self.char2id[self.TOKEN_END_OF_WORD] = char_id + 1

        self.id2char.append(self.TOKEN_BEGINING_OF_WORD)
        self.id2char.append(self.TOKEN_END_OF_WORD)

        self.dic_size = len(self.char2id)

        print(f"Character Dictionaries are built- Vocab size is {self.dic_size}")

        def encode_word(self,word):
            """
            :return the shape of the encoded word:[self.max-word-len + 2,]
            """










