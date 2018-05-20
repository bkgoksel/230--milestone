from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import FastText

class WordReplacer(object):
    def __init__(self, vec_file, n_replacements):
        self.model = FastText.load_fasttext_format(vec_file)
        self.lemmatizer = WordNetLemmatizer()
        self.n_replacements = n_replacements
        self.replace_dict = {}

    def get_replacements(self, word):
        """
        Get non-synonym similar replacements for the word:
            * Get N closest words from word vectors
            * Get the lemmas for the word and the closest words
            * Filter out the synonyms
        """
        if word in self.replace_dict:
            return self.replace_dict[word]
        replacements = []
        n_left = self.n_replacements - len(replacements)
        while n_left:
            closest, _ = zip(*(self.model.wv.similar_by_word(word, topn=n_left)))
            lemmas = {word: self.lemmatizer.lemmatize(word) for word in [word] + closest}
            synonyms = [l.name() for syn in wordnet.synsets(word) for l in syn.lemmas()]
            replacements += [w for w, l in lemmas.items() if l not in synonyms]
            n_left = self.n_replacements - len(replacements)
        reps = replacements[:self.n_replacements]
        self.replace_dict[word] = reps
        return reps
