# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downloads data.
"""
from collections import defaultdict
from re import findall, sub
import string
import os
import texar.torch as tx


translation_table = defaultdict(lambda: ' ', {ord(letter): letter for letter in string.ascii_letters})
translation_table[ord('.')] = ' . '
translation_table[ord('?')] = ' ? '
translation_table[ord(',')] = ' , '
translation_table[ord('!')] = ' ! '


def parse_sentence(text, max_length=None):
    # remove links
    text = sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
               '', text)
    # split sentences
    for sentence in findall(r'(".+")|([^.?!]+[.?!])', text):
        for match in sentence:
            if match:
                translated = sub(r' +', ' ', match.translate(translation_table).strip())
                if len(translated) > 2:
                    if max_length is None or len(translated.split()) < max_length:
                        yield translated


def main():
    """Entrypoint.
    """
    tx.data.maybe_download(
        urls='https://drive.google.com/file/d/'
             '1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing',
        path='./',
        filenames='yelp.zip',
        extract=True)
    os.remove('yelp.zip')
    os.rename('./data/yelp/sentiment.dev.labels', './data/yelp/dev.labels')
    os.rename('./data/yelp/sentiment.dev.text', './data/yelp/dev.text')
    os.rename('./data/yelp/sentiment.test.labels', './data/yelp/test.labels')
    os.rename('./data/yelp/sentiment.test.text', './data/yelp/test.text')
    os.rename('./data/yelp/sentiment.train.labels', './data/yelp/train.labels')
    os.rename('./data/yelp/sentiment.train.text', './data/yelp/train.text')


if __name__ == '__main__':
    main()
