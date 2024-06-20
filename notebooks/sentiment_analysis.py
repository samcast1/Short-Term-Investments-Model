import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load data and preprocess
file_path = '~/all_reviews.csv'
reviews_df = pd.read_csv(file_path, nrows=1000)

reviews_df = reviews_df.sample(frac=1).reset_index(drop=True)

# Drop rows with null values in 'review_text' or 'rating'
reviews_df = reviews_df.dropna(subset=['review_text', 'rating'])

# Map star rating to label
def map_star_to_label(star_rating):
    return star_rating - 1

reviews_df['label'] = reviews_df['rating'].apply(map_star_to_label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    reviews_df['review_text'],
    reviews_df['label'],
    test_size=0.3,
    random_state=42,
    stratify=reviews_df['label']
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Dataset class for handling encodings and labels
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to torch.long
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = SentimentDataset(train_encodings, y_train.tolist())
test_dataset = SentimentDataset(test_encodings, y_test.tolist())

# Model configuration
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_hidden_layers = 12
config.num_labels = 5

# Class weights calculation
class_counts = reviews_df['label'].value_counts()
total_samples = len(reviews_df)
num_labels = 5
class_weights = {i: total_samples / class_counts.get(i, 1) for i in range(num_labels)}

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights_tensor = torch.tensor([class_weights[i] for i in range(num_labels)], dtype=torch.float).to(device)

# Model initialization
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)

# Custom forward method for handling class weights
def forward(input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
    outputs = model.bert(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        token_type_ids=token_type_ids.to(device),
    )
    sequence_output = outputs[1]
    logits = model.classifier(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1).to(device))
    return (loss, logits) if loss is not None else logits

# Override model's forward method
model.forward = forward

# Training arguments
training_args_run1 = TrainingArguments(
    output_dir='./results/run1',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.005,
    logging_dir='./logs/run1',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Trainer initialization
trainer_run1 = Trainer(
    model=model,
    args=training_args_run1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Training and evaluation
print("Running model . . .")
output = trainer_run1.train()
eval_result = trainer_run1.evaluate()

print("Training complete. Evaluation results:")
print(eval_result)

# Print training metrics
metrics = output.metrics
print("Training metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Evaluate on the test set and print results
print("Test evaluation results:")
test_results = trainer_run1.predict(test_dataset)
test_metrics = compute_metrics(test_results)
for key, value in test_metrics.items():
    print(f"{key}: {value}")

# Optionally, you can load and view tensorboard logs for more insights
# %load_ext tensorboard
# %tensorboard --logdir=./logs/run1
