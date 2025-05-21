from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=42).to(device)
model.load_state_dict(torch.load("best_transformer_model.pth", map_location=device), strict=False)
model.eval()

# Label mapping 
label_map = {
    0: 'COMEDY', 1: 'IMPACT', 2: 'POLITICS', 3: 'WELLNESS', 4: 'SPORTS', 5: 'PARENTS', 
    6: 'HEALTHY LIVING', 7: 'TASTE', 8: 'THE WORLDPOST', 9: 'BLACK VOICES', 10: 'ENTERTAINMENT', 
    11: 'BUSINESS', 12: 'ARTS', 13: 'TECH', 14: 'WORLD NEWS', 15: 'FOOD & DRINK', 16: 'FIFTY', 
    17: 'WOMEN', 18: 'STYLE & BEAUTY', 19: 'MEDIA', 20: 'PARENTING', 21: 'STYLE', 22: 'WEIRD NEWS', 
    23: 'EDUCATION', 24: 'WORLDPOST', 25: 'HOME & LIVING', 26: 'QUEER VOICES', 27: 'RELIGION', 
    28: 'WEDDINGS', 29: 'CRIME', 30: 'MONEY', 31: 'TRAVEL', 32: 'ENVIRONMENT', 33: 'GOOD NEWS', 
    34: 'DIVORCE', 35: 'U.S. NEWS', 36: 'GREEN', 37: 'COLLEGE', 38: 'SCIENCE', 39: 'LATINO VOICES', 
    40: 'CULTURE & ARTS', 41: 'ARTS & CULTURE'
}


def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_class = torch.argmax(logits, dim=-1).item()
    return label_map.get(pred_class, "Unknown")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_text = request.form['text']
        prediction = predict(user_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
