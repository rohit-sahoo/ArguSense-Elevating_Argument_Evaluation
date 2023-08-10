import pandas as pd
import warnings, logging, torch
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.metrics import log_loss
import torch.nn.functional as F
import numpy as np

warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

model_nm = '../deberta'

df = pd.read_csv('/feedback-prize-effectiveness/train.csv')

test_df = pd.read_csv('/feedback-prize-effectiveness/train.csv')

isLength = len(df.discourse_id.unique()) == len(df)

tokz = AutoTokenizer.from_pretrained(model_nm)

sep = tokz.sep_token
print(sep)

df['inputs'] = df.discourse_type + sep + df.discourse_text

new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
df = df.replace(new_label)
df = df.rename(columns={"discourse_effectiveness": "label"})

ds = Dataset.from_pandas(df)


def tok_func(x): return tokz(x["inputs"], truncation=True)


inps = "discourse_text", "discourse_type"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps + ('inputs', 'discourse_id', 'essay_id'))

essay_ids = df.essay_id.unique()
np.random.seed(42)
np.random.shuffle(essay_ids)
essay_ids[:5]

val_prop = 0.2
val_sz = int(len(essay_ids) * val_prop)
val_essay_ids = essay_ids[:val_sz]

is_val = np.isin(df.essay_id, val_essay_ids)
idxs = np.arange(len(df))
val_idxs = idxs[is_val]
trn_idxs = idxs[~is_val]

dds = DatasetDict({"train": tok_ds.select(trn_idxs),
                   "test": tok_ds.select(val_idxs)})


def get_dds(df, train=True):
    ds = Dataset.from_pandas(df)
    to_remove = ['discourse_text', 'discourse_type', 'inputs', 'discourse_id', 'essay_id']
    tok_ds = ds.map(tok_func, batched=True, remove_columns=to_remove)
    if train:
        return DatasetDict({"train": tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})
    else:
        return tok_ds


lr, bs = 8e-5, 16
wd, epochs = 0.01, 1


def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}


def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
                             evaluation_strategy="epoch", per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs * 2,
                             num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=3)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=score)


trainer = get_trainer(dds)
trainer.train()

test_df = pd.read_csv('/feedback-prize-effectiveness/train.csv')

test_df['inputs'] = test_df.discourse_type + sep + test_df.discourse_text

test_ds = get_dds(test_df, train=False)

preds = F.softmax(torch.Tensor(trainer.predict(test_ds).predictions)).numpy().astype(float)
print(preds)
