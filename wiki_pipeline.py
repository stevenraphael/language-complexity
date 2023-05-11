import spacy
from nltk import Tree
from wordfreq import zipf_frequency
from depLength import depLength
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch
import math


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

def get_last_token_prob(tokens):
    truncated_tokens = tokens[:-1]
    last_token = tokens[-1]
    tokens_tensor = torch.tensor([truncated_tokens])

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0][0, -1]
    token_logit = predictions[last_token]
    #print(token_logit)
    #print(sum(predictions))
    predictions = (predictions)-token_logit
    predictions = np.exp(predictions)/sum(np.exp(predictions))
    token_logit = predictions[last_token]
    print(tokenizer.decode(last_token))
    print(token_logit.item())
    print('')
    return (token_logit).item()


def get_gpt2_score(text):
    indexed_tokens = tokenizer.encode(text)
    return math.prod(get_last_token_prob(indexed_tokens[:i]) for i in range(2, len(indexed_tokens)))/(len(indexed_tokens)-2)
    #return get_last_token_prob(indexed_tokens)


#print(get_sentence_score("The child likes to play."))

words = []
with open('1-1000.txt') as f:
    words = f.readlines()

words_100 = [w.strip() for w in set(words[:100])]
print(words_100)

punctuation = "~!@#$%^&*)_+-=[]}{|;:',<.>/?\"\\"

good_regex = "^[A-Za-z0-9'-(),.; ]+$"

def good_text(text):
    return re.match(good_regex, text) and len(re.findall("\(;",text)) == 0


def clean_punc(text):
    a = ""
    for i in range(len(text) - 1):
        if (not(text[i]==' ' and text[i-1]=='(')) and (text[i] != ' ' or text[i+1] not in punctuation):
            a+=text[i]
    if text[-1] != ' ':
        a+=text[-1]
    return a

import re
def getWords(text):
    return re.compile("[\w'-]+").findall(text.lower())

def unpunctuate(text):
    return ' '.join(getWords(text))

def freqscore(text):
    words = getWords(text)
    new_words = [w for w in words if w not in words_100]
    if len(new_words) == 0:
        return 0
    return sum(zipf_frequency(w, 'en') for w in new_words)/len(new_words)

def raw_freqscore(text):
    words = getWords(text)
    new_words = words
    if len(new_words) == 0:
        return 0
    return sum(zipf_frequency(w, 'en') for w in new_words)/len(new_words)



def sentence_dict(text):
    text = clean_punc(text)
    print(text)
    return {"freqscore": freqscore(text), "deplength": depLength.DepLength(text).sdl()[0], "gpt2_prob": get_gpt2_score(text)}

en_nlp = spacy.load('en_core_web_sm')




comparison_dataset = load_dataset("embedding-data/simple-wiki")


sentence_results = []
doc = en_nlp(comparison_dataset["train"][0]["set"][0])
print([str(x) for x in doc.sents])
print(comparison_dataset["train"][0]["set"])
#def get_sent_results(text):


for i in range(20):#len(comparison_dataset["train"])):
    sents = comparison_dataset["train"][i]["set"]
    if good_text(sents[0]) and good_text(sents[1]):
        sentence_results.append([sentence_dict(sents[0]), sentence_dict(sents[1])])
    else: 
        sentence_results.append(None)
print(sentence_results)

