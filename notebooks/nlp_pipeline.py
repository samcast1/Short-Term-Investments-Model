import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import optuna
import logging
from sklearn.utils import resample
import os

from optuna.pruners import MedianPruner



print("Loading data and preprocessing...")
# Load data and preprocess


# Load and preprocess data
file_path = 'all_reviews.csv'
reviews_df = pd.read_csv(file_path, nrows=20000)
reviews_df = reviews_df.sample(frac=1).reset_index(drop=True)
reviews_df = reviews_df.dropna(subset=['review_text', 'rating'])

num_threads = 8  # Number of CPU cores to use
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)


# Map star rating to label
def map_star_to_label(star_rating):
    if star_rating in [1, 2]:
        return 0  # 1-2 stars
    elif star_rating == 3:
        return 1  # 3 stars
    elif star_rating in [4, 5]:
        return 2  # 4-5 stars
    else:
        raise ValueError(f"Invalid star rating: {star_rating}")

reviews_df['label'] = reviews_df['rating'].apply(map_star_to_label)

# Calculate desired number of samples for each class
n_samples = len(reviews_df)
n_class_2 = int(0.50 * n_samples)  # 50% for 4-5 stars
n_class_1 = int(0.20 * n_samples)  # 20% for 3 stars
n_class_0 = int(0.30 * n_samples)  # 30% for 1-2 stars

# Separate classes
class_0 = reviews_df[reviews_df['label'] == 0]
class_1 = reviews_df[reviews_df['label'] == 1]
class_2 = reviews_df[reviews_df['label'] == 2]

# Resample classes
class_2_downsampled = resample(class_2, replace=False, n_samples=n_class_2)
class_1_upsampled = resample(class_1, replace=True, n_samples=n_class_1)
class_0_upsampled = resample(class_0, replace=True, n_samples=n_class_0)

# Combine resampled classes
resampled_df = pd.concat([class_2_downsampled, class_1_upsampled, class_0_upsampled]).sample(frac=1).reset_index(drop=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    resampled_df['review_text'],
    resampled_df['label'],
    test_size=0.3,
    stratify=resampled_df['label']
)


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


# Set up logging
optuna.logging.set_verbosity(optuna.logging.INFO)  # Set the verbosity level to INFO

# Optionally, configure a custom logger
logger = optuna.logging.get_logger("optuna")
logger.addHandler(logging.StreamHandler())

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 2, 4, step=1)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.75)
    attention_dropout = trial.suggest_float('attention_dropout', 0.05, 0.55)
    num_layers = trial.suggest_int('num_layers', 4 ,6, step=1)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    warmup_steps = trial.suggest_int('warmup_steps', 100, 1000, step=100)
    
    # Print hyperparameters
    print(f"Trial hyperparameters: \n"
          f"learning_rate={learning_rate}, \n"
          f"num_train_epochs={num_train_epochs}, \n"
          f"dropout_rate={dropout_rate}, \n"
          f"attention_dropout={attention_dropout}, \n"
          f"num_layers={num_layers}, \n"
          f"weight_decay={weight_decay}, \n"
          f"warmup_steps={warmup_steps}")
    # Configure model
    config = DistilBertConfig(
        num_labels=3,
        dropout=dropout_rate,
        attention_dropout=attention_dropout,
        num_hidden_layers=num_layers,
    )

    model = DistilBertForSequenceClassification(config)

    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=learning_rate,
        per_device_train_batch_size=60,
        per_device_eval_batch_size=120,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=50,  # Log training progress every 50 steps
        report_to = None,
        dataloader_num_workers=num_threads # Number of worker threads
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    for epoch in range(num_train_epochs):
        # Print hyperparameters
        print(f"Trial hyperparameters: \n"
            f"learning_rate={learning_rate}, \n"
            f"num_train_epochs={num_train_epochs}, \n"
            f"dropout_rate={dropout_rate}, \n"
            f"attention_dropout={attention_dropout}, \n"
            f"num_layers={num_layers}, \n"
            f"weight_decay={weight_decay}, \n"
            f"warmup_steps={warmup_steps}")
        trainer.train()
        eval_result = trainer.evaluate()
        trial.report(eval_result['eval_loss'], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    # trainer.train()
    # eval_result = trainer.evaluate()
    # return eval_result['eval_loss']

pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=100, interval_steps=20)
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=5, n_jobs=1)
