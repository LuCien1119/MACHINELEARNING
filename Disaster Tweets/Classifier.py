import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

for i in [df_train, df_test]:
    for j in ['keyword', 'location']:
        i[j] = i[j].fillna('')

def text_cleaner(df):
    df['text'] = df['text'].replace(r'http\S+|www.\S+', '', regex=True)
    df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    return df
text_cleaner(df_train)
text_cleaner(df_test)

df_train['text_cleaned'] = df_train['text'] + ' ' + df_train['keyword'] + ' ' + df_train['location']
df_test['text_cleaned'] = df_test['text'] + ' ' + df_test['keyword'] + ' ' + df_test['location']

text = df_train['text_cleaned'].values
text = text.tolist()
labels = df_train['target'].values
labels = labels.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
def tokenize(text):
    input = []
    for txt in text:
        txt_tokens = tokenizer.encode(txt,add_special_tokens=True, max_length=200 , padding= 'max_length', return_tensors='pt')
        input.append(txt_tokens)
    input = torch.cat(input, dim=0)
    return input

text_tokenized = tokenize(text)
labels = torch.tensor(labels)
dataset = TensorDataset(text_tokenized, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

def create_attention_mask(input_ids):
    return (input_ids != 0).long()

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, labels = tuple(t.to(device) for t in batch)
        attention_mask = create_attention_mask(input_ids)
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = tuple(t.to(device) for t in batch)
            attention_mask = create_attention_mask(input_ids)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            preds.extend(logits)
            true_labels.extend(label_ids)
    preds = np.argmax(preds, axis=1)
    f1 = f1_score(true_labels, preds)
    return f1

epochs = 6
for epoch in range(epochs):
    avg_train_loss = train(model, train_dataloader, optimizer)
    f1 = evaluate(model, val_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Training loss: {avg_train_loss:.3f}')
    print(f'Validation F1: {f1:.3f}')

test = df_test['text_cleaned'].values
test = test.tolist()
test_tokenized = tokenize(test)
test_dataset = TensorDataset(test_tokenized)
test_dataloader = DataLoader(test_dataset,batch_size=64, shuffle=False)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = create_attention_mask(input_ids)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)

all_preds = np.array(all_preds)
submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['target'] = all_preds
submission.to_csv('submission.csv', index=False)