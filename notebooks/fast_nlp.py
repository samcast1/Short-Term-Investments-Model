import pandas as pd
from sklearn.model_selection import train_test_split
import torch
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import os
import time

print("Loading data and preprocessing...")
# Load data and preprocess
file_path = '~/all_reviews.csv'
reviews_df = pd.read_csv(file_path, nrows=50)

reviews_df = reviews_df.sample(frac=1).reset_index(drop=True)

# Drop rows with null values in 'review_text' or 'rating'
reviews_df = reviews_df.dropna(subset=['review_text', 'rating'])

# Map star rating to label
def map_star_to_label(star_rating):
    return star_rating - 1

reviews_df['label'] = reviews_df['rating'].apply(map_star_to_label)

print("Splitting data into train and test sets...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    reviews_df['review_text'],
    reviews_df['label'],
    test_size=0.3,
    random_state=42,
    stratify=reviews_df['label']
)

torch.set_num_threads(torch.get_num_threads())

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_data(data, tokenizer):
    print("Tokenizing data...")
    start_time = time.time()
    encodings = tokenizer(data, truncation=True, padding='max_length', return_tensors='pt')
    end_time = time.time()
    print(f"Tokenization completed in {end_time - start_time:.2f} seconds.")
    return encodings

print("Tokenizing training data...")
train_encodings = tokenize_data(X_train.tolist(), tokenizer)
print("Tokenizing test data...")
test_encodings = tokenize_data(X_test.tolist(), tokenizer)



# Dataset class for handling encodings and labels
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

print("Creating datasets...")
# Create datasets
train_dataset = SentimentDataset(train_encodings, y_train.tolist())
test_dataset = SentimentDataset(test_encodings, y_test.tolist())


# Model configuration
print("Calculating class weights...")
class_counts = reviews_df['label'].value_counts()
total_samples = len(reviews_df)
num_labels = 5
class_weights_tensor = torch.tensor([total_samples / class_counts.get(i, 1) for i in range(num_labels)], dtype=torch.float)
class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()

config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=5)
config.num_hidden_layers = 5


# Custom model class for handling class weights
class CustomDistilBERTForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.class_weights = class_weights  # Store class weights as an attribute

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Using the CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

# Model initialization
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomDistilBERTForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config, class_weights=class_weights_tensor)

# Training arguments

import multiprocessing
num_cpus = multiprocessing.cpu_count()
print('Number of cpus for training: ', num_cpus)
training_args_run1 = TrainingArguments(
    output_dir='./results/run1',
    num_train_epochs=3,
    per_device_train_batch_size=90,  # Increased batch size
    per_device_eval_batch_size=120,  # Increased batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs/run1',
    logging_steps=10,  # Reduced logging frequency
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed-precision training
    dataloader_num_workers=num_cpus,  # Number of worker threads
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

print("Initializing Trainer...")
# Trainer initialization

metrics_dir = os.path.expanduser("~/metrics")
os.makedirs(metrics_dir, exist_ok=True)

trainer_run1 = Trainer(
    model=model,
    args=training_args_run1,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    output_dir=metrics_dir
)

# Training and evaluation
print("Running model training...")
output = trainer_run1.train()
print("Evaluating model...")
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
