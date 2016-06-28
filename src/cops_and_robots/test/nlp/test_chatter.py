class TestChatter(object):
    """docstring for TestChatter"""
    def __init__(self, arg):
        super(TestChatter, self).__init__()
        self.arg = arg
        

    def test_NLP_pipeline():

        # Grab data, remember indices of training/test data #######################
        dh = DataHandler()

        # Find ids associated with each sentence group
        j = 0
        sentence_ids = []
        for i, row in dh.df.iterrows():
            if isinstance(row['Input Sentence'], str):
                j += 1
            sentence_ids.append(j)

        # Split corpus into training and test sets
        corpus = dh.input_sentences
        corpus = [[i,s] for i,s in enumerate(dh.input_sentences)]
        random.shuffle(corpus)
        n = len(corpus) // 2
        training_corpus = corpus[:n]
        test_corpus = corpus[n:]
        training_ids = [d[0] for d in training_corpus]

        # Tokenize test corpus (unigram model)
        input_document = "\n".join([d[1] for d in test_corpus])
        tokenizer = Tokenizer(max_distance=1)
        tokenized_test_corpus = tokenizer.tokenize((input_document))[0]

        # Get Sierra's and Jeremy's trained taggers ###############################

        # Grab the full training data
        sierra_full_data = zip(dh.df["Sierra's Tokens"].tolist(),
                               dh.df["Sierra's Labels"].tolist())
        jeremy_full_data = zip(dh.df["Jeremy's Tokens"].tolist(),
                               dh.df["Jeremy's Labels"].tolist())

        # Limit J&S's training data to the randomized corpus
        sierra_td = [d for i, d in enumerate(sierra_full_data)
                     if sentence_ids[i] in training_ids]
        jeremy_td = [d for i, d in enumerate(jeremy_full_data)
                     if sentence_ids[i] in training_ids]

        # Ignore NaN lines
        sierra_td = [d for d in sierra_td if not (isinstance(d[0], float)
                    or isinstance(d[1], float))]
        jeremy_td = [d for d in jeremy_td if not (isinstance(d[0], float)
                    or isinstance(d[1], float))]

        # Prepare training files and paths
        data_dir = os.path.dirname(__file__) + '/data/'
        sierra_training_file = "sierra_training.txt"
        jeremy_training_file = "jeremy_training.txt"
        sierra_training_path = data_dir + sierra_training_file
        jeremy_training_path = data_dir + jeremy_training_file

        # Write training files
        for tf, td in zip([sierra_training_path, jeremy_training_path],
                          [sierra_td, jeremy_td]):
            with open(tf,'w') as file_:
                for d in td:
                    str_ = '\t'.join(d) + '\n'
                    str_ = str_.replace (" ", "_")
                    file_.write(str_)

        # Train tagging engine using both Sierra's and Jeremy's tags
        s_tagger = Tagger(training_file='sierra_training.txt',
                          test_file='sierra_test.txt',
                          input_file='sierra_input.txt',
                          model_file='sierra_model.txt',
                          output_file='sierra_output.txt',
                          )
        j_tagger = Tagger(training_file='jeremy_training.txt',
                          test_file='jeremy_test.txt',
                          input_file='jeremy_input.txt',
                          model_file='jeremy_model.txt',
                          output_file='jeremy_output.txt',
                          )

        # Evaluate tagging agreement ##############################################
        s_tagged_document = s_tagger.tag_document(tokenized_test_corpus)
        j_tagged_document = j_tagger.tag_document(tokenized_test_corpus)

        agreements = []
        for i, _ in enumerate(s_tagged_document):
            agreement = s_tagged_document[i][1] == j_tagged_document[i][1]
            if agreement:
                agreements.append(1)
            else:
                agreements.append(0)
        agreements = np.array(agreements)

        print agreements.mean()

        # Generate TDCs ###########################################################
        # s_TDCs = TDC_Collection(s_tagged_document)
        # j_TDCs = TDC_Collection(j_tagged_document)
        # j_TDCs.plot_TDCs()
