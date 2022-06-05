# Text Style Transfer

## Usage
```python
from model_wrapper import TextStyleTransfer

style_transfer = TextStyleTransfer('config')
original_text = "This is an example text to test text style transfer"
print(f'Original text: {original_text}')
print(f'Transferred text: {style_transfer.transfer(original_text)}')
```
