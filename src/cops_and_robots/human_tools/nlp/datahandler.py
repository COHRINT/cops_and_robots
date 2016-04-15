class DataHandler(object):
    """docstring for DataHandler"""

    def __init__(self, data_url=''):
        # Get data from google sheet associated with a specific URL
        if len(data_url) < 1:
            self.url = 'https://docs.google.com/spreadsheet/ccc?key=1S-nYlSuQGCbfUTTw2DKebJ6xZJRmya3RklXC9MpjbBk&gid=2089135022&output=csv'
        else:
            self.url = data_url

        data = requests.get(self.url).content

        # Find the input sentences
        self.df = pd.read_csv(StringIO(data))
        self.input_sentences = [a for a in self.df['Input Sentence']
                                if isinstance(a, str)]

        # Find the proper single word tokenization
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

        # Merge SWT with the main dataframe
        try:
            del self.df['Approx. Single Word Tokens']
        except KeyError:
            logging.debug("No column to delete")
        self.df["Single Word Tokens"] = pd.Series(single_word_tokens,
                                                  index=self.df.index)
        cols = self.df.columns.tolist()
        cols = [cols[0]] + [cols[-1]] + cols[1:-1]
        self.df = self.df[cols]