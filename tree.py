import spacy
from nltk import Tree
from wordfreq import zipf_frequency
from depLength import depLength


#https://pypi.org/project/wordfreq/
#https://pypi.org/project/PassivePy/


#https://stackoverflow.com/questions/7633274/extracting-words-from-a-string-removing-punctuation-and-returning-a-list-with-s
import re
def getWords(text):
    return re.compile('\w+').findall(text)

en_nlp = spacy.load('en_core_web_sm')

def freqscore(text):
    words = getWords(text)
    return sum(zipf_frequency(w, 'en') for w in words)/len(words)



doc = en_nlp("As the Universe cooled after the big bang it eventually became possible for particles as we know them to exist. ")

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


s1 = """Linguistics is the scientific study of human language."""
s2 = """Linguistics is the study of language. People who study language are called linguists."""
print(sum(len(getWords(str(sent))) for sent in doc.sents)/sum(1 for _ in doc.sents))
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
print(freqscore(s1))
print(freqscore(s2))
print(depLength.DepLength(s1).sdl())
print(depLength.DepLength(s2).sdl())

