import numpy as np


class Metrics:
    def __init__(self):
        super(Metrics, self).__init__()

    def _levenshtein_distance(self, ref, hyp):
        """
        :param ref: First sequence and or sentence (sequence of words)
        :param hyp: Second sequence and or sentence (sequence of words)
        :return: the edit distance between both sequences
        """
        m = len(ref)
        n = len(hyp)

        if ref == hyp:
            return 0
        if m == 0:
            return n
        if n == 0:
            return m

        if m < n:
            ref, hyp = hyp, ref
            m, n = n, m

        distance = np.zeros((2, n + 1), dtype=np.int32)

        for j in range(0, n + 1):
            distance[0][j] = j

        for i in range(1, m + 1):
            prev_row_idx = (i - 1) % 2
            cur_row_idx = i % 2
            distance[cur_row_idx][0] = i
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
                else:
                    s_num = distance[prev_row_idx][j - 1] + 1
                    i_num = distance[cur_row_idx][j - 1] + 1
                    d_num = distance[prev_row_idx][j] + 1
                    distance[cur_row_idx][j] = min(s_num, i_num, d_num)
        return distance[m % 2][n]

    def avg_wer(self, wer_scores, combined_ref_len):
        return float(sum(wer_scores)) / float(combined_ref_len)

    def word_errors(self, reference, hypothesis, ignore_case=False, delimiter=' '):
        """Compute the levenshtein distance between reference sequence and
        hypothesis sequence in word-level.
        :param reference: The reference sentence.
        :param hypothesis: The hypothesis sentence.
        :param ignore_case: Whether case-sensitive or not.
        :param delimiter: Delimiter of input sentences.
        :return: Levenshtein distance and word number of reference sentence.
        """
        if ignore_case:
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        ref_words = reference.split(delimiter)
        hyp_words = hypothesis.split(delimiter)

        edit_distance = self._levenshtein_distance(ref_words, hyp_words)
        return float(edit_distance), len(ref_words)

    def char_errors(self, reference, hypothesis, ignore_case=False, remove_space=False):
        """Compute the levenshtein distance between reference sequence and
        hypothesis sequence in char-level.
        :param reference: The reference sentence.
        :param hypothesis: The hypothesis sentence.
        :param ignore_case: Whether case-sensitive or not.
        :param remove_space: Whether remove internal space characters
        :return: Levenshtein distance and length of reference sentence.
        """
        if ignore_case:
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        join_char = ' '
        if remove_space:
            join_char = ''

        reference = join_char.join(filter(None, reference.split(' ')))
        hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

        edit_distance = self._levenshtein_distance(reference, hypothesis)
        return float(edit_distance), len(reference)

    def wer(self, reference, hypothesis, ignore_case=False, delimiter=' '):
        """Calculate word error rate (WER). WER compares reference text and
        hypothesis text in word-level.
        :param reference: The reference sentence.
        :param hypothesis: The hypothesis sentence.
        :param ignore_case: Whether case-sensitive or not.
        :param delimiter: Delimiter of input sentences.
        :return: Word error rate.
        """
        edit_distance, ref_len = self.word_errors(reference, hypothesis, ignore_case,
                                                  delimiter)

        if ref_len == 0:
            raise ValueError("Length of the reference should be > 0.")

        wer = float(edit_distance) / ref_len
        return wer

    def cer(self, reference, hypothesis, ignore_case=False, remove_space=False):
        """Calculate charactor error rate (CER). CER compares reference text and
        hypothesis text in char-level. CER is defined as:
        :param reference: The reference sentence.
        :param hypothesis: The hypothesis sentence.
        :param ignore_case: Whether case-sensitive or not.
        :param remove_space: Whether remove internal space characters
        :return: Character error rate.
        :raises ValueError: If the reference length is zero.
        """
        edit_distance, ref_len = self.char_errors(reference, hypothesis, ignore_case,
                                                  remove_space)

        if ref_len == 0:
            raise ValueError("Length of reference should be > 0.")

        cer = float(edit_distance) / ref_len
        return cer
