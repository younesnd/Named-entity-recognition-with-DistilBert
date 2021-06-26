from flask import Flask,render_template,url_for,request
from transformers import AutoTokenizer, DistilBertConfig, DistilBertForTokenClassification
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

tag_value = ['I-art','B-art','I-per','I-geo','I-org','B-org','I-tim','B-eve','B-geo','B-per','O','B-gpe','I-gpe','B-nat','I-eve','B-tim','I-nat', 'PAD']
tag_values = dict.fromkeys(tag_value).keys()
tag_values= list(tag_values)
tag2idx = {t: i for i, t in enumerate(tag_values)}
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=True)
model = torch.load(r"/home/younesnd/Downloads/best_model.pt",  map_location='cpu')
model.eval()
app = Flask(__name__)
@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process',methods=["POST"])
def process():
  text = request.form['content']
  tokenized_sentence = tokenizer.encode(text)
  input_ids = torch.tensor([tokenized_sentence])
  with torch.no_grad():
	  output = model(input_ids)
  label_indices = np.argmax(output[0].numpy(), axis=2)
  
  tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
  new_tokens, new_labels = [] , []
  for token, label_idx in zip(tokens, label_indices[0]):
	  if token.startswith("##"):
		  new_tokens[-1] = new_tokens[-1] + token[2:]
	  else:
		  new_labels.append(tag_values[label_idx])
		  new_tokens.append(token)
  list_Tokens,list_labels= [],[]
  for  token ,label in zip(new_tokens,new_labels):
      list_Tokens.append(token)
      list_labels.append(label)
  res = {} 
  for key in list_Tokens: 
	  for value in list_labels: 
		  res[key] = value 
		  list_labels.remove(value) 
		  break 
  res.pop('[CLS]', None)
  res.pop('[SEP]', None)
  for key, value in res.items():
    if res[key]=="I-per":
        res[key] = "I-Person"
    elif res[key]=="B-per":
        res[key] = "B-Person"
    elif res[key]=="O":
        res[key] = "Outside"
    elif res[key]=="B-org":
        res[key] = "B-Organization"
    elif res[key]=="I-org":
        res[key] = "I-Organization"
    elif res[key]=="B-geo":
        res[key] = "B-Geographical Entity"
    elif res[key]=="B-gpe":
        res[key] = "B-Geopolitical Entity"
    elif res[key]=="I-geo":
        res[key] = "I-Geographical Entity"
    elif res[key]=="I-gpe":
        res[key] = "I-Geopolitical Entity"
    elif res[key]=="B-eve":
        res[key] = "B-Event"
    elif res[key]=="I-eve":
        res[key] = "I-Event"
    elif res[key]=="B-tim":
        res[key] = "B-Time indicator"
    elif res[key]=="I-tim":
        res[key] = "I-Time indicator"
    elif res[key]=="I-art":
        res[key] = "I-Artifact"
    elif res[key]=="B-art":
        res[key] = "B-Artifact"
    elif res[key]=="B-nat":
        res[key] = "B-Natural Phenomenon"
    elif res[key]=="I-nat":
        res[key] = "I-Natural Phenomenon"
  
  
  
  return render_template("index.html", results= res )


if __name__ == '__main__':
	app.run(debug=True)
