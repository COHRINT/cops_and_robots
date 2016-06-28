import logging
from StringIO import StringIO  # got moved to io in python3.
import requests
import re
import pandas as pd
import spacy.en
import spacy.parts_of_speech
from collections import Counter
import string


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
        return self.corpus

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

    def get_multi_word_tokens(self, merge_tokens=False):
        tokens = {}
        tokens['jeremy'] = self.df["Jeremy's Tokens"].tolist()
        tokens['sierra'] = self.df["Sierra's Tokens"].tolist()
        
        tokens['sierra'] = [d for d in tokens['sierra'] if not isinstance(d, float)]
        tokens['jeremy'] = [d for d in tokens['jeremy'] if not isinstance(d, float)]

        return tokens['jeremy']

    def get_selected_statements(self):
        statements = {}
        statements['jeremy'] = self.df["Jeremy's Selected Template"].tolist()
        statements['sierra'] = self.df["Sierra's Selected Template"].tolist()
        
        statements['sierra'] = [d for d in statements['sierra'] if not isinstance(d, float)]
        statements['jeremy'] = [d for d in statements['jeremy'] if not isinstance(d, float)]

        return statements

    def separate_two_words(self, tokens):
        two_list=[]
        for item in tokens:
            #print len(item.split())
            if len(item.split()) == 2:
                two_list.append(item)
        return two_list

    def separate_three_words(self, tokens):
        three_list=[]
        for item in tokens:
            #print len(item.split())
            if len(item.split()) == 3:
                three_list.append(item)
        return three_list

    def POS(self, token_list):
        nlp = spacy.en.English()
        list_pos=[]
        combo = ""
        for item in token_list:
            doc = nlp(unicode(item), tag = True, parse =True)
            for tok in doc:
                combo = combo + str(tok.pos_) + " "
            list_pos.append(combo)
            combo = ""
        return list_pos

    def split_input_into_two_words(self):
        singly_tokenized_document = re.findall(r"[\w']+|[.,!?;]", self.corpus)
        full_list=[]
        for i,word_one in enumerate(singly_tokenized_document[:-1]):
            full_list.append(word_one + " " + singly_tokenized_document[i+1])
        return full_list

    def split_input_into_three_words(self):
        singly_tokenized_document = re.findall(r"[\w']+|[.,!?;]", self.corpus)
        full_list=[]
        for i,word_one in enumerate(singly_tokenized_document[:-2]):
            full_list.append(word_one + " " + singly_tokenized_document[i+1]+" " + singly_tokenized_document[i+2])
        return full_list

    def graph(self, full_keys, full_values, new_value_list):
        import matplotlib.pyplot as plt
        import numpy as np

        width = .8

        z=zip(map(int,full_values),map(int,new_value_list),full_keys)
        z=sorted(z, reverse=True)
        full_values,new_value_list,full_keys=zip(*z[:50])
        N=len(full_values)
        ind = np.arange(N)
        p1 = plt.bar(ind, full_values, width, color='black')
        p2 = plt.bar(ind, new_value_list, width, color='mediumspringgreen')

        plt.ylabel('Scores')
        plt.title('Three word tokenizations')
        plt.xticks(ind + width/2.,(full_keys),rotation=80)
        plt.yticks(np.arange(0,max(full_values)+20,10))
        plt.legend((p1[0],p2[0]),('Full list','tokenized list'))
        plt.show()

    def count_em_up(self, data):
        valuables=Counter(data)
        keys=[]
        values=[]
        for key,value in valuables.iteritems():
            keys.append(str(key))
            values.append(str(value))
        return keys, values

    def compare(self, token_list, token_values, full_list, full_values):
        new_value_list=[0]*len(full_list)
        for i,word in enumerate(full_list):
            if word in token_list:
                    new_value_list[i]=token_values[token_list.index(word)] 
        return new_value_list


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    dh = DataHandler()

    # Two words
    full_data = dh.split_input_into_two_words()
    full_two_pos = dh.POS(full_data)
    full_keys_2,full_values_2 = dh.count_em_up(full_two_pos)
    
    data_to_parse = dh.get_multi_word_tokens()
    two_words = dh.separate_two_words(data_to_parse)
    two_pos=dh.POS(two_words)
    two_keys,two_values = dh.count_em_up(two_pos)

    new_value_list_2 = dh.compare(two_keys,two_values,full_keys_2,full_values_2)
    dh.graph(full_keys_2, full_values_2, new_value_list_2)

    # #three words
    full_data = dh.split_input_into_three_words()
    full_three_pos = dh.POS(full_data)
    full_three_pos = dh.separate_three_words(full_three_pos)
    full_keys_3,full_values_3 = dh.count_em_up(full_three_pos)
    
    three_words = dh.separate_three_words(data_to_parse)
    three_pos=dh.POS(three_words)
    three_keys,three_values = dh.count_em_up(three_pos)

    new_value_list_3 = dh.compare(three_keys,three_values,full_keys_3,full_values_3)
    dh.graph(full_keys_3, full_values_3, new_value_list_3)