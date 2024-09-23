import re, random, pickle, argparse
from typing import Tuple


class NotInVocabError(Exception):
    pass



def load_corpus(path: str) -> str:
    """
    Load and preprocess the text file corpus
    :param path: Path to the file, including filename and extension
    :return: the loaded and pre-processed text corpus as a string
    """
    with open(path, 'r') as tales:
        # read in all text
        all_text = tales.read()
        # replace newline characters with empty space
        all_text = all_text.strip()  # replace('\n', ' ').strip()
        # remove any words that are in all CAPS
        all_text = re.sub(r'\b[A-Z]{2,}\b', '', all_text)
        # remove some non-word characters
        all_text = re.sub(r'[^a-zA-Z0-9\s.]', '', all_text)
        # replace multiple white space with single whitespace
        all_text = re.sub(r'\s+', ' ', all_text)
        # put spaces around the periods so that they are split in future steps
        all_text = all_text.replace('.', ' . ')
        # replace all uppercase characters with lowercase characters
        all_text = all_text.lower()

    return all_text


class NGram:
    def __init__(self, n: int, model_path: str, load: bool = False):
        """
        Initialize a WordPredictor class
        :param n: Order of the n-gram
        :param model_path: Path to where model will be or has been saved (.p extension for Pickle file)
        :param load: Whether or not to load the model from the specified path
        """
        self.n: int = n
        self.model_path: str = model_path
        # load model if requested, otherwise initialize empty word dictionary
        if load:
            self._load()
        else:
            # this will hold the word prediction model
            self.words = {}

    def train(self, corpus: str):
        """
        Train the model on a string of text data
        :param corpus: string of text data
        :return: None
        """
        # split words out from string by whitespace (this creates unigrams)
        all_words = corpus.split(' ')
        # single_words = all_words.copy()
        # we need to create n-grams
        if self.n > 1:
            all_words = [" ".join(all_words[i:i + self.n]) for i in range(len(all_words))[:-self.n + 1]]

        # loop through the corpus
        for index, word in enumerate(all_words):
            # if we have not encountered this word yet, add it to the dictionary
            if word not in self.words:
                self.words[word] = {"next_words": {}, "next_words_count": 0}
            # tally the word that comes after this one
            if index < len(all_words) - 1:
                next_word = all_words[index + 1].split(" ")[-1]
                if next_word in self.words[word]["next_words"]:
                    self.words[word]["next_words"][next_word] += 1
                else:
                    self.words[word]["next_words"][next_word] = 1
                self.words[word]["next_words_count"] += 1

        # calculate next word probabilities for every word in the dictionary
        for word in self.words:
            next_words_count = self.words[word]["next_words_count"]
            choices = list(self.words[word]["next_words"].keys())
            probabilities = []
            for next_word in self.words[word]["next_words"]:
                probabilities.append(self.words[word]["next_words"][next_word] / next_words_count)
            self.words[word]["choices"] = choices
            self.words[word]["probabilities"] = probabilities

    def predict_next_word(self, word: Tuple, deterministic: bool=False):
        """
        Predict the next word given the current word (or words)
        :param word: current word (or words)
        :param deterministic: Boolean flag that if True, samples the highest probabilty word, if False,
            randomly samples from the probability distribution
        :return: next word
        """
        if word not in self.words:
            raise NotInVocabError(f"The word(s), {word}, not in the vocabulary.")
        choices = self.words[word]["choices"]
        probabilities = self.words[word]["probabilities"]
        if deterministic:
            return self.words[word]["choices"][max(range(len(choices)), key=lambda x: probabilities[x])]
        else:
            return random.choices(choices, weights=probabilities, k=1)[0]

    def save(self):
        """
        Save the word dictionary to a file using pickle
        :return: None
        """
        # serialize the trained WordPredictor object using pickle and dump it to a file
        with open(self.model_path, 'wb') as pfile:
            pickle.dump(self.words, pfile)

    def _load(self):
        """
        Load the word dictionary from a file
        :return: None
        """
        # load the words dictionary from pickle file
        with open(self.model_path, 'rb') as pfile:
            self.words = pickle.load(pfile)


class BytePairEncoding:
    """
    Class for the Byte-Pair Encoding (BPE) tokenizer algorithm
    """
    def __init__(self, model_path: str, load: bool = False):
        """
        Initialize a BytePairEncoding class
        :param model_path: Path to where model will be or has been saved (.p extension for Pickle file)
        :param load: Whether or not to load the model from the specified path
        """
        self.model_path = model_path
        # load vocabulary if requested, otherwise initialize empty vocabulary
        if load:
            self._load()
        else:
            # this will hold the vocabulary
            self.vocabulary = {}

    def train(self, corpus: str, num_iter: int, verbose: bool=False):
        """
        Train the model
        :param corpus: a string containing training text
        :param num_iter: number of iterations to run the BPE training loop
        :param verbose: True to enable printing of training progress
        :return: None
        """
        # Initialize vocabulary with unique characters in corpus
        self.vocabulary = dict.fromkeys([c for c in corpus])
        # Loop for as many iterations as requested
        for i in range(num_iter):
            # Determine most probable pair of tokens
            mpp = self._most_probable_pair(corpus)
            if verbose:
                print(i, mpp)
            # stop if there are no more pairs that occur more than once
            if mpp is None:
                break
            else:
                # add a new token to the vocabulary that is the concatenation/merging of the
                # most probably pairs
                self.vocabulary[''.join(mpp)] = None
                # replace all occurences of the pairs with the new merged token in the text
                corpus = self._replace_pair(corpus, mpp)
        # convert our vocabulary to a list
        self.vocabulary = list(self.vocabulary.keys())

    def _most_probable_pair(self, corpus):
        """
        Get the most probable/most frequent token pair in the provided corpus
        :param corpus: A corpus of text, as a string
        :return: A tuple of two tokens
        """
        pairs = {}
        # loop through all tokens in the corpus
        for i in range(len(corpus) - 1):
            current_token = corpus[i]
            next_token = corpus[i+1]
            # this is the pair at the current index
            pair = (current_token, next_token)
            # count the number of occurrences of this pair
            if pair not in pairs:
                pairs[pair] = 1
            else:
                pairs[pair] += 1
        # sort the token pairs by number of occurrences
        sorted_pairs = sorted(pairs.items(), key=lambda item: item[1], reverse=True)
        if sorted_pairs[0][1] == 1:
            # if no token pair occurs more than once, return None
            return None
        else:
            # return most frequent token pair
            return sorted_pairs[0][0]

    def _replace_pair(self, corpus, pair):
        """
        Replace a pair of tokens in the corpus with the merged/concatenated version of said pair.
        We use this for tokenization in inference mode
        :param corpus: Tokenized text (list of tokens) in which to perform replacement
        :param pair: Pair of tokens formatted as a Tuple of strings
        :return: text with replacement
        """
        # this will hold the new list of tokens
        result = []
        i = 0
        # find all occurrences of the pair
        while i < len(corpus) - 1:
            # if we encounter the pair of tokens, replace it with the merged pair
            if (corpus[i], corpus[i + 1]) == pair:
                result.append(''.join(pair))
                i += 2
            else:
                # if this isn't a match, simply append the current token to the new tokenized text
                result.append(corpus[i])
                i += 1

        return result

    def _replace_token(self, tokenized_string, token):
        """
        Replace
        :param string:
        :param token:
        :return:
        """
        result = []
        i = 0
        # loop through the tokenized string and replace all pairs of consecutive tokens that
        # equal the provided token when merged, with that token
        while i < len(tokenized_string) - 1:
            # check if the merged pair of tokens equal the provided token, and if so, replace it with the
            # provided token
            if ''.join((tokenized_string[i], tokenized_string[i + 1])) == token:
                result.append(token)
                i += 2
            else:
                result.append(tokenized_string[i])
                i += 1
        if i == len(tokenized_string) - 1:
            result.append(tokenized_string[i])

        return result

    def tokenize(self, s: str):
        """
        Tokenize a string of text using the learned vocabulary
        :param s: String of text
        :return: Tuple(list of tokens, list of token IDs)
        """
        # split our string into individual characters
        tokenized = [c for c in s]
        # loop through each token in our vocabulary
        for token in self.vocabulary:
            if len(token) > 1:
                # search for every occurrence of consecutive tokens in the tokenized string, that when combined,
                # equal the current token from our vocabulary, and replace it with that token
                # For example, if the token from our vocabulary is 'he' and the current tokenized text is
                # ['h', 'e', 'l', 'l', 'o'], the output would be ['he', 'l', 'l', 'o']
                tokenized = self._replace_token(tokenized, token)

        return tokenized, self.tokens_to_ids(tokenized)

    def tokens_to_ids(self, tokens):
        """
        Convert tokens to IDs

        :param tokens: list of tokens
        :return: list of token IDs
        """
        token_ids = []
        for token in tokens:
            # in this implementation, the token IDs are just the index of the token in the vocabulary
            token_ids.append(self.vocabulary.index(token))
        return token_ids

    def save(self):
        """
        Save the vocabulary to a file using pickle
        :return: None
        """
        # serialize the trained vocabulary object using pickle and dump it to a file
        with open(self.model_path, 'wb') as pfile:
            pickle.dump(self.vocabulary, pfile)

    def _load(self):
        """
        Load the vocabulary from a file
        :return: None
        """
        # load the vocabulary from pickle file
        with open(self.model_path, 'rb') as pfile:
            self.vocabulary = pickle.load(pfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("N-gram Word Predictor")
    parser.add_argument('a', type=str, help="Activity to perform",
                        choices=["train_ngram", "predict_ngram", "train_bpe", "tokenize"])
    parser.add_argument('--data', type=str, default='./grimm_fairy_tales.txt',
                        help="path to text corpus, including filename and extension")
    parser.add_argument('--save', type=str, default="./MyModel.p",
                        help="path to where model will be saved, including filename and extension (.p)")
    parser.add_argument('--load', type=str, default="./MyModel.p",
                        help="path to where pickle model to be loaded, including filename and extension")
    parser.add_argument('--word', type=str, default='the', help="first word to start sequence prediction")
    parser.add_argument('--nwords', type=int, default=100, help='number of words to generate')
    parser.add_argument('--text', type=str, help="specifies the string to be tokenized through the 'tokenize' activity.")
    parser.add_argument('--n', type=int, help='order of the n-gram for train_ngram or predict_ngram activity')
    parser.add_argument('--d', action='store_true', help='enables deterministic sampling')
    parser.add_argument('--k', type=int, default=500, help="number of iterations of BPE training loops")
    args = parser.parse_args()

    if args.a == "train_ngram":
        # Load and pre-process text corpus
        all_text = load_corpus(args.data)

        # Create a WordPredictor instance and train it on the text corpus
        ngram = NGram(args.n, args.save, False)
        ngram.train(all_text)

        # save the trained model
        ngram.save()

    if args.a == "predict_ngram":
        # Load the saved word predictor
        wp = NGram(args.n, args.load, load=True)

        # make the requested starting word lowercase
        word = args.word.lower()
        print(word, end=' ')

        # let's do 10 words per line
        num_words_generated = 0
        for _ in range(args.nwords // 10 + 1):
            for _ in range(10 if args.nwords - num_words_generated >= 10 else args.nwords - num_words_generated):
                previous_word = word
                # predict and print the next word
                word = wp.predict_next_word(word, args.d)
                print(word, end=' ')
                word = " ".join(previous_word.split(" ")[-args.n+1:]) + " " + word
                num_words_generated += 1
            # print a newline
            print()

    if args.a == "train_bpe":
        with open(args.data, 'r') as f:
            training_text = f.read()
            # training_text = add_eow_character(training_text)
        bpe = BytePairEncoding(args.save)
        bpe.train(training_text, args.k, verbose=True)
        bpe.save()
        print("Done")

    if args.a == "tokenize":
        bpe = BytePairEncoding(args.load, load=True)
        # print(bpe.vocabulary)
        tokens, token_ids = bpe.tokenize(args.text)
        print("Tokens:    ", tokens)
        print("Token IDs: ", token_ids)