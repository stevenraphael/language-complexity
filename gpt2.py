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


def get_sentence_score(text):
    indexed_tokens = tokenizer.encode(text)
    return math.prod(get_last_token_prob(indexed_tokens[:i]) for i in range(2, len(indexed_tokens)))/(len(indexed_tokens)-2)
    #return get_last_token_prob(indexed_tokens)


print(get_sentence_score("Linguistics is the study of language."))
#print(get_sentence_score("The child likes to play."))