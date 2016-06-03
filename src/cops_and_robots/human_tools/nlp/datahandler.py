import logging
from StringIO import StringIO  # got moved to io in python3.
import requests
import re
import pandas as pd


class DataHandler(object):
    """docstring for DataHandler"""

    def __init__(self, data_url=''):
        self.load_from_url(data_url)
        self.get_input_sentences()

    def load_from_url(self, data_url):
        # Get data from google sheet associated with a specific URL
        if len(data_url) < 1:
            self.url = 'https://docs.google.com/spreadsheet/ccc?key=1S-nYlSuQGCbfUTTw2DKebJ6xZJRmya3RklXC9MpjbBk&gid=2089135022&output=csv'
        else:
            self.url = data_url

        data = requests.get(self.url).content
        self.df = pd.read_csv(StringIO(data))

    def get_input_sentences(self):
        self.input_sentences = [a for a in self.df['Input Sentence']
                                if isinstance(a, str)]
        self.corpus = " ".join(self.input_sentences)

    def get_single_word_tokens(self, add_to_df=True):
        single_word_tokens = [re.findall(r"[\w']+|[.,!?;]", a)
                              for a in self.input_sentences]
        single_word_tokens = [b for a in single_word_tokens for b in a]

        # Tokenize punctuation as well
        punct = [".",",","!","?",";"]
        n = len(single_word_tokens); i = 0
        while i < n - 1:
            token = single_word_tokens[i]
            next_token = single_word_tokens[i + 1]
            
            if token in punct and next_token in punct:
                del single_word_tokens[i + 1]
                n -= 1
            else:
                i += 1
        self.single_word_tokens = single_word_tokens

        # Merge SWT with the main dataframe
        if add_to_df:
            try:
                del self.df['Approx. Single Word Tokens']
            except KeyError:
                logging.debug("No column to delete")
            self.df["Single Word Tokens"] = pd.Series(single_word_tokens,
                                                      index=self.df.index)
            cols = self.df.columns.tolist()
            cols = [cols[0]] + [cols[-1]] + cols[1:-1]
            self.df = self.df[cols]

        return single_word_tokens

    def get_multi_word_tokens(self, add_to_df=True):
        single_word_tokens = [re.findall(r"[\w']+|[.,!?;]", a)
                              for a in self.input_sentences]
        single_word_tokens = [b for a in single_word_tokens for b in a]

        # Tokenize punctuation as well
        punct = [".",",","!","?",";"]
        n = len(single_word_tokens); i = 0
        while i < n - 1:
            token = single_word_tokens[i]
            next_token = single_word_tokens[i + 1]
            
            if token in punct and next_token in punct:
                del single_word_tokens[i + 1]
                n -= 1
            else:
                i += 1
        self.single_word_tokens = single_word_tokens

        # Merge SWT with the main dataframe
        if add_to_df:
            try:
                del self.df['Approx. Single Word Tokens']
            except KeyError:
                logging.debug("No column to delete")
            self.df["Single Word Tokens"] = pd.Series(single_word_tokens,
                                                      index=self.df.index)
            cols = self.df.columns.tolist()
            cols = [cols[0]] + [cols[-1]] + cols[1:-1]
            self.df = self.df[cols]

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dh = DataHandler()
    print dh.df
    # dh.get_single_word_tokens()
    print dh.single_word_tokens