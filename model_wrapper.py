from importlib import import_module
import os.path as path
import texar.torch as tx
import torch
from ctrl_gen_model import CtrlGenModel
from prepare_data import parse_sentence


def text2ids(text_tokens, vocab, input_len=0):
    if text_tokens[0] != vocab.bos_token:
        text_tokens.insert(0, vocab.bos_token)
    if text_tokens[-1] != vocab.eos_token and text_tokens[-1] != '':
        text_tokens.append(vocab.eos_token)
    if len(text_tokens) < input_len+1:
        for _ in range(input_len-len(text_tokens)+1):
            text_tokens.append('')
    text_ids = vocab.map_tokens_to_ids_py(text_tokens)
    return text_ids


class TextStyleTransfer:
    def __init__(self, config_module, datasets_default_path="./data",
                 device='cpu', checkpoint_name='final_model.pth'):
        device = torch.device(device)
        config = import_module(config_module)
        config = tx.HParams(config.model, None)
        checkpoint_path = path.join(config.checkpoint_path, checkpoint_name)
        assert path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.input_len = checkpoint['input_len']

        # FIXME: should not access that default hparams
        vocab_hp = tx.HParams({'vocab_file': f"{datasets_default_path}/{checkpoint['dataset']}/vocab"},
                              tx.data.data.multi_aligned_data._default_dataset_hparams())
        self.vocab = tx.data.MultiAlignedData.make_vocab([vocab_hp])[0]
        self.model = CtrlGenModel(self.input_len,
                                  self.vocab, config, device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def transfer(self, text, transfer_clas=None):
        # Eval
        text += ' .'
        results = []
        for sentence in parse_sentence(text):
            text_tokens = sentence.split()
            text_ids = text2ids(text_tokens, self.vocab, self.input_len)
            output_ids = self.model.infer(text_ids=text_ids, transfer_clas)

            hyps = self.vocab.map_ids_to_tokens_py(output_ids)
            output_str = ' '.join(hyps.tolist())
            results.append(output_str)
        return results

    def classify(self, text):
        text += ' .'
        class_counter = {}
        for sentence in parse_sentence(text):
            text_tokens = sentence.split()
            text_ids = text2ids(text_tokens, self.vocab, self.input_len)
            clas = self.model.classify(text_ids)
            class_counter[clas] = class_counter[clas]+1 if clas in class_counter else 1
        return max(class_counter, key=class_counter.get)


# Testing
if __name__ == "__main__":
    personality_transfer = TextStyleTransfer('config')
    original_text = "Japonica is a Japanese, Sushi restaurant, with excellent food quality and decent decor. Dojo, which is a Japanese, Vegetarian restaurant, with decent food quality, has mediocre decor."
    print(f'Original text: {original_text}')
    print(f'Transferred text: {personality_transfer.transfer(original_text, transfer_clas=0)}')
