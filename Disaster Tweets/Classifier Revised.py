import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

for i in [df_train,df_test]:
    for j in ['keyword','location', 'text']:
        i[j] = i[j].fillna('')
# fill the nulls

def text_clean(text):
    text['text'] = text['text'].replace(r'http\S+|www.\S+', '', regex=True)
    text['text'] = text['text'].str.replace(r'[^\w\s]', '', regex=True)
    return text
text_clean(df_train)
text_clean(df_test)
df_train["text_clean"] = df_train["text"] + " " + df_train["keyword"] + " " + df_train["location"]
df_test["text_clean"] = df_test["text"] + " " + df_test["keyword"] + " " + df_test["location"]
# print(df_train.head())
# eliminate noises in column 'text'

X_train, X_val, y_train, y_val = train_test_split(df_train["text_clean"].values,
                                                  df_train['target'].values,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify= df_train.target.values
                                                  )

tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=True
                                          )

encoded_train = tokenizer.batch_encode_plus(X_train,
                          add_special_tokens=True,
                          return_attention_mask=True,
                          padding='max_length',
                          max_length=256,
                          return_tensors='pt'
                          )

encoded_val = tokenizer.batch_encode_plus(X_val,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        padding='max_length',
                        max_length=256,
                        return_tensors='pt'
                        )
# tonkenize training set

labels_train = torch.tensor(y_train)
input_ids_train = encoded_train['input_ids']
attention_mask_train = encoded_train['attention_mask']

labels_val = torch.tensor(y_val)
input_ids_val = encoded_val['input_ids']
input_attention_mask_val = encoded_val['attention_mask']

dataset_train = TensorDataset(input_ids_train,
                              attention_mask_train,
                              labels_train
                              )

dataset_val = TensorDataset(input_ids_val,
                            input_attention_mask_val,
                            labels_val)

batch_size = 4

train_loader = DataLoader(dataset_train,
                          sampler=RandomSampler(dataset_train),
                          batch_size=batch_size
                          )

val_loader = DataLoader(dataset_val,
                        sampler=RandomSampler(dataset_val),
                        batch_size=32
                        )
# set batch size of training set

model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                      num_labels=df_train.target.unique().size,
                                                      output_attentions=False,
                                                      output_hidden_states=False,
                                                      )
epochs = 4

optimizer = AdamW(
    model.parameters(),
    lr=1e-5,
    eps=1e-8
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader)*epochs
)

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# set random seed to ensure recurrence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# set GPU as device

def f1_score_func(y_preds, y_true):
    y_preds_flat = np.argmax(y_preds, axis=1).flatten()
    y_true_flat = y_true.flatten()
    return f1_score(y_true_flat, y_preds_flat, average='weighted')

def evaluate(val_loader):
    model.eval()

    loss_val_total = 0
    y_pred, y_true = [], []

    for batch in val_loader:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        y_pred.append(logits)
        y_true.append(label_ids)

    loss_val_avg = loss_val_total / len(val_loader)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return loss_val_avg, y_pred, y_true

for epoch in range(1, epochs+1):
    model.train()

    loss_train_total = 0

    progress_bar = tqdm(train_loader,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
                  }

        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # set the step of gradient calculation
        optimizer.step()
        scheduler.step()

        # progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(train_loader)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, y_pred, y_true = evaluate(train_loader)
    val_f1 = f1_score_func(y_pred, y_true)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score: {val_f1}')

X_test = df_test["text_clean"]
encoded_test = tokenizer.batch_encode_plus(X_test,
                          add_special_tokens=True,
                          return_attention_mask=True,
                          padding='max_length',
                          max_length=256,
                          return_tensors='pt'
                          )
dataset_test = TensorDataset(encoded_test['input_ids'],
                             encoded_test['attention_mask']
                             )
test_loader = DataLoader(dataset_test,
                         batch_size=64,
                         shuffle=False
                         )

model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1]
                  }
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)

all_preds = np.array(all_preds)
submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['target'] = all_preds
submission.to_csv('submission.csv', index=False)