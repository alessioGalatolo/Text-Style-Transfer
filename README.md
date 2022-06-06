# Text Style Transfer
Replication of the paper [Unsupervised Text Style Transfer using Language Models as Discriminators](https://arxiv.org/abs/1805.11749). 

PyTorch port of [Texar/Text Style Transfer](https://github.com/asyml/texar/tree/master/examples/text_style_transfer)

## Requirements
Mainly PyTorch, Texar. See requirements.txt

## Usage (replication)
Download yelp dataset
```bash
python prepare_data.py
```
Train the model
```bash
python train.py
```
### Custom dataset
You will need to create your own preprocessing of your custom dataset. The function `parse_sentence` in `prepare_data.py` may be helpful for this. 

NB: when using a custom dataset you need to change `max_decoding_length_train` and `max_decoding_length_infer` in `config.py` to match the length of your sentence. Or you need to limit the length: if you are using `parse_sentece` you can do `parse_sentence("some text", max_length=20)`. This is the recommended option as the original authors report better results on shorter sentences.

Put your dataset under `data/your_dataset_name`. The dataset should contain train/dev/test .text and .labels files alongside the vocab file (you can use texar utils for this: see `tx.data.make_vocab`) 

Train the model with `python train.py --dataset your_dataset_name`. The model will be saved under `checkpoints/final_model.pth`.

Integrate the model in your pipeline:

```python
from model_wrapper import TextStyleTransfer

style_transfer = TextStyleTransfer('config')
original_text = "This is an example text to test text style transfer"
print(f'Original text: {original_text}')
print(f'Transferred text: {style_transfer.transfer(original_text)}')
```
