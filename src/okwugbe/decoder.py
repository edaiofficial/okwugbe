import torch


class Decoders:
    def __init__(self):
        super(Decoders, self).__init__()

    def greedy_decoder(self, text_transform, output, labels, label_lengths, collapse_repeated=True):
        blank_label = text_transform.get_blank_index()
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
        return decodes, targets
