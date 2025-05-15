import os
import time

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find(
        "vader_lexicon.zip"
    )  # Corrected to check for the zip file often downloaded
except LookupError:
    nltk.download("vader_lexicon")


# --- Model Configuration ---
def get_sentence_transformer_embeddings(sentences, model_name_or_path, batch_size):
    print(f"Loading SentenceTransformer model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)
    print(f"Generating embeddings with {model_name_or_path}...")
    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    print(
        f"Embeddings with {model_name_or_path} generated in {time.time() - start_time:.2f} seconds"
    )
    return embeddings


def get_huggingface_auto_model_embeddings(
    sentences, model_name_or_path, batch_size, max_length=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading HuggingFace AutoModel: {model_name_or_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)

    if isinstance(sentences, (np.ndarray, pd.Series)):
        sentences = sentences.tolist()

    all_embeddings = []
    print(f"Generating embeddings with {model_name_or_path}...")
    start_time = time.time()
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            all_embeddings.append(
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )  # CLS token
    print(
        f"Embeddings with {model_name_or_path} generated in {time.time() - start_time:.2f} seconds"
    )
    return np.vstack(all_embeddings)


def extract_sentiment_features_func(
    sentences, batch_size=None
):  # batch_size is unused but kept for consistent signature
    print("Extracting sentiment-specific features (VADER, TextBlob)...")
    start_time = time.time()
    sid = SentimentIntensityAnalyzer()
    features_list = []
    for sentence in sentences:
        vader_scores = sid.polarity_scores(sentence)
        blob = TextBlob(sentence)
        feature_vector = [
            vader_scores["pos"],
            vader_scores["neg"],
            vader_scores["neu"],
            vader_scores["compound"],
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            sentence.count("!"),
            sentence.count("?"),
        ]
        features_list.append(feature_vector)
    print(f"Sentiment features extracted in {time.time() - start_time:.2f} seconds")
    return np.array(features_list)


ALL_AVAILABLE_MODELS = {
    "mpnet": {
        "name": "mpnet",
        "embedding_function": get_sentence_transformer_embeddings,
        "embedding_dim": 768,
        "file_key": "mpnet",
        "model_name_or_path": "all-mpnet-base-v2",
        "processor_output_dim": 512,
        "kwargs": {},
    },
    "sentiment_distilbert": {
        "name": "sentiment_distilbert",
        "embedding_function": get_sentence_transformer_embeddings,
        "embedding_dim": 768,
        "file_key": "sentiment_distilbert",
        "model_name_or_path": "distilbert-base-uncased-finetuned-sst-2-english",
        "processor_output_dim": 512,
        "kwargs": {},
    },
    "sentiment_features": {
        "name": "sentiment_features",
        "embedding_function": extract_sentiment_features_func,
        "embedding_dim": 8,  # VADER (4) + TextBlob (2) + Punctuation (2)
        "file_key": "sentiment_features",
        "processor_output_dim": 64,
        "kwargs": {},
    },
    "twitter_roberta": {
        "name": "twitter_roberta",
        "embedding_function": get_huggingface_auto_model_embeddings,
        "embedding_dim": 768,  # For 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        "file_key": "twitter_roberta",
        "model_name_or_path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "processor_output_dim": 512,
        "kwargs": {"max_length": 128},  # Example of passing specific kwarg
    },
    # ── 1️⃣  ADD THIS BLOCK ───────────────────────────────────────────────────────────
    "deberta_v3": {
        "name": "deberta_v3",  # unique key
        "embedding_function": get_huggingface_auto_model_embeddings,
        "embedding_dim": 768,  # hidden size of DeBERTa‑V3‑base
        "file_key": "deberta_v3",  # filename prefix for .npz cache
        "model_name_or_path": "microsoft/deberta-v3-base",  # HF model ID
        "processor_output_dim": 512,  # keep in line with others
        "kwargs": {"max_length": 128},  # same truncation you use
    },
    # ──────────────────────────────────────────────────────────────────────────────────
    # Add more models here following the same structure
    # e.g. "my_custom_bert": { "name": "my_custom_bert", "embedding_function": ..., ...}
}

# --- User Selection: Choose which models to use ---
MODELS_TO_USE = [
    "sentiment_distilbert",
    # "mpnet",
    # "twitter_roberta",
    # "sentiment_features",
    # "deberta_v3",
]  # Example: Use these three
# MODELS_TO_USE = ["mpnet", "sentiment_features"] # Example: Use only two
# MODELS_TO_USE = list(ALL_AVAILABLE_MODELS.keys()) # Example: Use all available models

active_models_config = [
    ALL_AVAILABLE_MODELS[name] for name in MODELS_TO_USE if name in ALL_AVAILABLE_MODELS
]
active_model_names = [model["name"] for model in active_models_config]
if not active_models_config:
    raise ValueError("MODELS_TO_USE list is empty or contains no valid model names.")
print(f"Selected models for this run: {active_model_names}")

# Constants
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 7  # Adjust as needed
PATIENCE = 5
RANDOM_SEED = 42
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Check if all required embeddings exist
embedding_files_exist = True
for model_config in active_models_config:
    if not os.path.exists(
        os.path.join(EMBEDDING_DIR, f"{model_config['file_key']}_embeddings.npz")
    ):
        embedding_files_exist = False
        break

# 1. Load data
print("Loading data...")
training_data = pd.read_csv("training.csv", index_col=0)  # Ensure you have these files
test_data = pd.read_csv("test.csv", index_col=0)  # Ensure you have these files

# 2. Prepare labels
label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
reverse_mapping = {-1: "negative", 0: "neutral", 1: "positive"}
training_data["label_encoded"] = training_data["label"].map(label_mapping)

# 3. Split data
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    training_data["sentence"].values,
    training_data["label_encoded"].values,
    test_size=0.1,
    stratify=training_data["label_encoded"],
    random_state=RANDOM_SEED,
)

# 4. Generate or load embeddings
train_embeddings_dict = {}
val_embeddings_dict = {}
test_embeddings_dict = {}

if embedding_files_exist:
    print("Loading pre-computed embeddings for selected models...")
    for model_config in active_models_config:
        model_name = model_config["name"]
        file_path = os.path.join(
            EMBEDDING_DIR, f"{model_config['file_key']}_embeddings.npz"
        )
        data = np.load(file_path)
        train_embeddings_dict[model_name] = data["train_embeddings"]
        val_embeddings_dict[model_name] = data["val_embeddings"]
        test_embeddings_dict[model_name] = data["test_embeddings"]
        print(
            f"Loaded {model_name} embeddings: Train {train_embeddings_dict[model_name].shape}, Val {val_embeddings_dict[model_name].shape}, Test {test_embeddings_dict[model_name].shape}"
        )
else:
    print("Generating embeddings for selected models...")
    for model_config in active_models_config:
        model_name = model_config["name"]
        print(f"\n--- Generating for Expert: {model_name} ---")
        emb_func = model_config["embedding_function"]

        # Pass model_name_or_path if the function expects it (handled by kwargs)
        func_kwargs = {"batch_size": BATCH_SIZE}
        if "model_name_or_path" in model_config:
            func_kwargs["model_name_or_path"] = model_config["model_name_or_path"]
        func_kwargs.update(model_config.get("kwargs", {}))

        train_embeddings_dict[model_name] = emb_func(train_sentences, **func_kwargs)
        val_embeddings_dict[model_name] = emb_func(val_sentences, **func_kwargs)
        test_embeddings_dict[model_name] = emb_func(
            test_data["sentence"].values, **func_kwargs
        )

        file_path = os.path.join(
            EMBEDDING_DIR, f"{model_config['file_key']}_embeddings.npz"
        )
        np.savez_compressed(
            file_path,
            train_embeddings=train_embeddings_dict[model_name],
            val_embeddings=val_embeddings_dict[model_name],
            test_embeddings=test_embeddings_dict[model_name],
        )
        print(f"Saved {model_name} embeddings to {file_path}")

    print("\nAll selected embeddings generated and saved.")


# 5. Define the Dynamic Mixture of Experts model
class SentimentMoE(nn.Module):
    def __init__(self, experts_config_list):
        super(SentimentMoE, self).__init__()
        self.experts_config = experts_config_list
        self.expert_names = [expert["name"] for expert in experts_config_list]

        self.processors = nn.ModuleDict()
        total_gate_input_dim = 0
        total_fusion_input_dim = 0

        for expert_config in self.experts_config:
            name = expert_config["name"]
            input_dim = expert_config["embedding_dim"]
            processor_output_dim = expert_config["processor_output_dim"]

            self.processors[name] = nn.Sequential(
                nn.Linear(input_dim, processor_output_dim),
                nn.ReLU(),
                nn.Dropout(
                    0.3 if processor_output_dim > 64 else 0.2
                ),  # Smaller dropout for smaller layers
            )
            total_gate_input_dim += input_dim
            total_fusion_input_dim += processor_output_dim

        self.gate = nn.Sequential(
            nn.Linear(total_gate_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.experts_config)),
            nn.Softmax(dim=1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_fusion_input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 3),  # 3 sentiment classes
        )

    def forward(self, expert_inputs_dict):
        processed_outputs = []
        gate_concat_inputs = []

        # Ensure order for concatenation matches self.expert_names used in training data prep
        for name in self.expert_names:
            expert_emb = expert_inputs_dict[name]
            processed_outputs.append(self.processors[name](expert_emb))
            gate_concat_inputs.append(expert_emb)  # Use original embeddings for gating

        combined_gate_input = torch.cat(gate_concat_inputs, dim=1)
        expert_weights = self.gate(combined_gate_input)

        weighted_expert_outputs = []
        for i, name in enumerate(self.expert_names):
            weighted_expert_outputs.append(
                processed_outputs[i] * expert_weights[:, i].unsqueeze(1)
            )

        combined_fusion_input = torch.cat(weighted_expert_outputs, dim=1)
        logits = self.fusion(combined_fusion_input)
        return logits, expert_weights


# 6. Setup training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_labels_adjusted = np.array([l + 1 for l in train_labels])
val_labels_adjusted = np.array([l + 1 for l in val_labels])

model = SentimentMoE(active_models_config).to(device)

# Create TensorDatasets dynamically
train_tensors = [
    torch.FloatTensor(train_embeddings_dict[name]) for name in active_model_names
]
train_tensors.append(torch.LongTensor(train_labels_adjusted))
train_dataset = TensorDataset(*train_tensors)

val_tensors = [
    torch.FloatTensor(val_embeddings_dict[name]) for name in active_model_names
]
val_tensors.append(torch.LongTensor(val_labels_adjusted))
val_dataset = TensorDataset(*val_tensors)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)

class_counts = np.bincount(train_labels_adjusted)
# Handle cases where a class might be missing (especially with small datasets/splits)
if len(class_counts) < 3:  # Assuming 3 classes, 0, 1, 2 after adjustment
    full_class_counts = np.zeros(3, dtype=int)
    full_class_counts[: len(class_counts)] = class_counts
    class_counts = full_class_counts
# Avoid division by zero if a class has zero samples (though stratification should prevent this for train)
weights = [1.0 / count if count > 0 else 1.0 for count in class_counts]
class_weights = torch.FloatTensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

# 7. Training loop
best_val_score = 0.0
patience_counter = 0
num_active_experts = len(active_model_names)

print("\nStarting training:")
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    expert_weights_sum_train = torch.zeros(num_active_experts).to(device)
    samples_count_train = 0

    for batch_idx, batch_data in enumerate(train_loader):
        labels = batch_data[-1].to(device)
        input_embeddings_list = [data.to(device) for data in batch_data[:-1]]

        # Create dict for model input
        expert_inputs_dict_train = {
            name: emb for name, emb in zip(active_model_names, input_embeddings_list)
        }

        optimizer.zero_grad()
        logits, expert_weights = model(expert_inputs_dict_train)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        expert_weights_sum_train += expert_weights.sum(dim=0).detach()
        samples_count_train += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_expert_weights_train = expert_weights_sum_train / samples_count_train

    model.eval()
    val_loss = 0.0
    all_preds_val = []
    all_labels_val = []
    expert_weights_sum_val = torch.zeros(num_active_experts).to(device)
    samples_count_val = 0

    with torch.no_grad():
        for batch_data in val_loader:
            labels = batch_data[-1].to(device)
            input_embeddings_list = [data.to(device) for data in batch_data[:-1]]
            expert_inputs_dict_val = {
                name: emb
                for name, emb in zip(active_model_names, input_embeddings_list)
            }

            logits, expert_weights = model(expert_inputs_dict_val)
            loss = criterion(logits, labels)
            _, predicted = torch.max(logits, 1)

            val_loss += loss.item()
            all_preds_val.extend(
                (predicted.cpu().numpy() - 1).tolist()
            )  # Back to -1, 0, 1
            all_labels_val.extend(
                (labels.cpu().numpy() - 1).tolist()
            )  # Back to -1, 0, 1
            expert_weights_sum_val += expert_weights.sum(dim=0).detach()
            samples_count_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    avg_expert_weights_val = expert_weights_sum_val / samples_count_val

    mae_val = mean_absolute_error(all_labels_val, all_preds_val)
    val_score = 0.5 * (2 - mae_val)  # Custom score, higher is better
    scheduler.step(val_score)

    conf_matrix = confusion_matrix(all_labels_val, all_preds_val, labels=[-1, 0, 1])
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Time: {epoch_time:.2f}s")
    print(
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Score: {val_score:.4f}"
    )

    train_weights_str = ", ".join(
        [
            f"{name} {avg_expert_weights_train[i]:.3f}"
            for i, name in enumerate(active_model_names)
        ]
    )
    print(f"Train Expert Weights: {train_weights_str}")
    val_weights_str = ", ".join(
        [
            f"{name} {avg_expert_weights_val[i]:.3f}"
            for i, name in enumerate(active_model_names)
        ]
    )
    print(f"Val Expert Weights: {val_weights_str}")
    print("Confusion Matrix (Val):")
    print(conf_matrix)

    if val_score > best_val_score:
        best_val_score = val_score
        patience_counter = 0
        torch.save(model.state_dict(), "best_sentiment_moe_model.pt")
        with open("best_model_metadata.txt", "w") as f:
            f.write(f"epoch: {epoch + 1}\n")
            f.write(f"val_score: {best_val_score:.4f}\n")
            f.write(f"active_models: {','.join(active_model_names)}\n")
            for i, name in enumerate(active_model_names):
                f.write(f"{name}_weight: {avg_expert_weights_val[i].item():.4f}\n")
        print(f"Model saved with score: {best_val_score:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# 8. Load best model for inference
try:
    model.load_state_dict(torch.load("best_sentiment_moe_model.pt"))
    print("Loaded best model state for inference.")
    try:
        metadata = {}
        with open("best_model_metadata.txt", "r") as f:
            for line in f:
                key, value = line.strip().split(": ", 1)
                metadata[key] = value
        print(
            f"Loaded best model from epoch {metadata.get('epoch', 'unknown')} with validation score {metadata.get('val_score', 'unknown')}"
        )
        print(
            f"Model was trained with experts: {metadata.get('active_models', 'unknown')}"
        )
        loaded_weights_str = ", ".join(
            [
                f"{name} {metadata.get(f'{name}_weight', '?')}"
                for name in active_model_names
            ]
        )
        print(f"Expert weights from metadata: {loaded_weights_str}")

    except FileNotFoundError:
        print("Best model metadata file not found.")
    except Exception as e:
        print(f"Error reading metadata: {e}")

except FileNotFoundError:
    print(
        "No best model checkpoint found. Using current model state (likely from last epoch of training if training just finished)."
    )
except Exception as e:
    print(f"Error loading model: {e}. Using current model state.")


# 9. Generate predictions on test data
model.eval()
test_preds = []
test_expert_weights_sum = torch.zeros(num_active_experts).to(device)
test_samples_count = 0

print("Generating predictions for test data...")
start_time = time.time()

# Prepare test tensors once
test_input_tensors_full = {
    name: torch.FloatTensor(test_embeddings_dict[name]).to(device)
    for name in active_model_names
}

# 9. Generate predictions on test data  (fixed)
model.eval()
test_preds = []
test_expert_weights_sum = torch.zeros(num_active_experts).to(device)
test_samples_count = 0

print("Generating predictions for test data...")
start_time = time.time()

for i in range(0, len(test_data), TEST_BATCH_SIZE):
    end = i + TEST_BATCH_SIZE

    # 🔹  make the per-batch dict of tensors
    batch_expert_inputs_dict = {
        name: test_input_tensors_full[name][i:end]  # slice cached embeddings
        for name in active_model_names
    }
    current_batch_size = batch_expert_inputs_dict[active_model_names[0]].size(0)

    with torch.no_grad():
        logits, expert_weights = model(batch_expert_inputs_dict)
        _, predicted = torch.max(logits, 1)  # 0,1,2
        test_preds.extend((predicted.cpu().numpy() - 1).tolist())  # → –1,0,1

        test_expert_weights_sum += expert_weights.sum(dim=0).detach()
        test_samples_count += current_batch_size

avg_test_expert_weights = test_expert_weights_sum / test_samples_count
print(f"Test inference done in {time.time() - start_time:.2f}s")
print(
    "Average expert weights on test:",
    {n: f"{w:.3f}" for n, w in zip(active_model_names, avg_test_expert_weights)},
)

# 10. Save submission -----------------------------------------------------------
submission = pd.DataFrame(
    {"id": test_data.index, "label": [reverse_mapping[p] for p in test_preds]}
)
submission.to_csv("submission.csv", index=False)
print("✅  submission.csv written")
