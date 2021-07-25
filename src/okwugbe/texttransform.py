import unicodedata

accent_code = [b'\\u0301', b'\\u0300', b'\\u0306', b'\\u0308', b'\\u0303']
alpha = {'ɔ': 0, 'ɛ': 5}
accents = {b'\\u0301': 1, b'\\u0300': 2, b'\\u0306': 3, b'\\u0308': 4, b'\\u0303': 5}
mapping = {1: 'ɔ́', 2: 'ɔ̀', 3: 'ɔ̆', 6: 'έ', 7: 'ὲ', 8: 'ɛ̆'}


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self, path):
        special_chars = ["'", "<SPACE>", ".", ","]
        with open(path, "r", encoding="utf-8") as fh:
            char_map_str = fh.read()
        chars_ = char_map_str.strip().split('\n')
        if chars_ == ['']:
            raise ValueError(
                "Length of the unique characters set should be > 0. Expecting a .txt file with one character per line")

        all_chars = special_chars + chars_
        self.char_map = {}
        self.index_map = {}
        for index, char in enumerate(all_chars):
            if char == '<SPACE>':
                ch = ' '
                self.char_map[ch] = int(index)
                self.index_map[int(index)] = ch
            else:
                self.char_map[char] = int(index)
                self.index_map[int(index)] = char

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        text = unicodedata.normalize("NFC", text)
        for c in self.get_better_mapping(text):
            try:
                if c == ' ':
                    ch = self.char_map['<SPACE>']
                elif c == '̀':
                    ch = 0
                else:
                    ch = self.char_map[c]
            except KeyError:
                # print("Error for character {} in this sentence: {}".format(c, text))
                ch = 0
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

    # we are following the idea that the composition gives the letter first followed by the sign(accent)
    # Chris Emezue
    def get_better_mapping(self, text):
        t_arr = [t for t in text]
        s = []
        for i in range(len(t_arr)):
            if t_arr[i].encode("unicode_escape") in accent_code:
                to_check = s[-1]
                try:
                    val = mapping[alpha[to_check] + accents[t_arr[i].encode("unicode_escape")]]
                    s.pop()
                    s.append(val)
                except KeyError:
                    print("Could not find for {} in sentence {} | Proceeding with default.".format(t_arr[i], text))
            else:
                s.append(t_arr[i])
        return s
