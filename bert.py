from transformers import AutoTokenizer, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


#https://stackoverflow.com/questions/61787853/how-to-get-the-probability-of-a-particular-tokenword-in-a-sentence-given-the-c
tok = AutoTokenizer.from_pretrained("bert-base-cased")
bert = BertForMaskedLM.from_pretrained("bert-base-cased")

input_idx = tok.encode(f"DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic {tok.mask_token} of organisms.")
#input_idx = tok.encode(f"The {tok.mask_token} were the best rock band ever.")
logits = bert(torch.tensor([input_idx]))[0]
prediction = logits[0].argmax(dim=1)
print(prediction)
print(tok.convert_ids_to_tokens(prediction[-6].numpy().tolist()))
print(tok.convert_ids_to_tokens(prediction[-5].numpy().tolist()))
print(tok.convert_ids_to_tokens(prediction[-4].numpy().tolist()))
print(tok.convert_ids_to_tokens(prediction[-3].numpy().tolist()))

#https://huggingface.co/transformers/v2.3.0/quickstart.html



# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic code of"
indexed_tokens = tokenizer.encode(text)
print(indexed_tokens)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
#tokens_tensor = tokens_tensor.to('cuda')
#model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word (in our case, the word 'man')
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)
print([tokenizer.decode([x]) for x in indexed_tokens])
#assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'