import torch


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    DIGITS = '0123456789'
    PUNCTUATION = '!"\'()*+,-./:;<=>?[\\]_{}|’№'
    CYRILLIC = 'ЁАБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё'
    LATIN = 'ABCDEFGHIJKLMNOPRSTVWYabcdefghiklmnoprstuvwxyz'
    ALPHABET = DIGITS + PUNCTUATION + CYRILLIC + LATIN
    OOV_TOKEN = '<OOV>'
    CTC_BLANK = '<BLANK>'

    def __init__(self, custom_alphabet=None):
        alphabet = self.ALPHABET if custom_alphabet is None else custom_alphabet
        self.char_map = self.get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def get_char_map(self, alphabet):
        """Returns a dictionary with characters as keys and int as values.
        Add BLANK char for CTC loss and OOV char for out of vocabulary symbols."""
        char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        char_map[self.CTC_BLANK] = 0
        char_map[self.OOV_TOKEN] = 1
        return char_map

    def encode_word(self, word):
        """Returns encoded word (int)."""
        return [self.char_map[char]
                if char in self.char_map
                else self.char_map[self.OOV_TOKEN]
                for char in word]

    def encode_list(self, word_list):
        """Returns a list of encoded words (int)."""
        return [self.encode_word(word) for word in word_list]

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list, merge_repeated=True):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary tokens."""
        return [self.decode_word(enc_word, merge_repeated)
                for enc_word in enc_word_list]

    def decode_word(self, enc_word, merge_repeated=True):
        """Returns a word (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary tokens."""
        return ''.join([self.rev_char_map[char_enc]
                        for idx, char_enc in enumerate(enc_word)
                        if (
                                not self.is_OOV_TOKEN(char_enc)
                                and not self.is_CTC_BLANK(char_enc)
                                and not (merge_repeated and idx > 0 and char_enc == enc_word[idx - 1])
                        )])

    def is_OOV_TOKEN(self, char_enc):
        return char_enc == self.char_map[self.OOV_TOKEN]

    def is_CTC_BLANK(self, char_enc):
        return char_enc == self.char_map[self.CTC_BLANK]


class BestPathDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decode(self, output):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary tokens."""
        pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
        return self.tokenizer.decode(pred)
