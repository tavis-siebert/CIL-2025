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

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

# Constants
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 7
PATIENCE = 5
RANDOM_SEED = 42
EMBEDDING_DIM = 768  # Base embedding dimension for mpnet

# Check if embeddings already exist
embeddings_dir = "embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

# Create embedding file paths for each expert
mpnet_file = os.path.join(embeddings_dir, "mpnet_embeddings.npz")
sentiment_file = os.path.join(embeddings_dir, "sentiment_embeddings.npz")
enhanced_file = os.path.join(embeddings_dir, "enhanced_features.npz")

# Flag to check if all embeddings exist
all_embeddings_exist = (
    os.path.exists(mpnet_file)
    and os.path.exists(sentiment_file)
    and os.path.exists(enhanced_file)
)

# 1. Load data
print("Loading data...")
training_data = pd.read_csv("training.csv", index_col=0)
test_data = pd.read_csv("test.csv", index_col=0)

# 2. Prepare labels
label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
reverse_mapping = {-1: "negative", 0: "neutral", 1: "positive"}
training_data["label_encoded"] = training_data["label"].map(label_mapping)

# 3. Split data with stratification
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    training_data["sentence"].values,
    training_data["label_encoded"].values,
    test_size=0.1,
    stratify=training_data["label_encoded"],
    random_state=RANDOM_SEED,
)


# Function to extract sentiment-specific features
def extract_sentiment_features(sentences):
    sid = SentimentIntensityAnalyzer()
    features = []

    for sentence in sentences:
        # Get VADER scores
        vader_scores = sid.polarity_scores(sentence)

        # Get TextBlob sentiment
        blob = TextBlob(sentence)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity

        # Count exclamation and question marks
        exclamation_count = sentence.count("!")
        question_count = sentence.count("?")

        # Combine all features
        feature_vector = [
            vader_scores["pos"],
            vader_scores["neg"],
            vader_scores["neu"],
            vader_scores["compound"],
            textblob_polarity,
            textblob_subjectivity,
            exclamation_count,
            question_count,
        ]

        features.append(feature_vector)

    return np.array(features)


# 4. Generate or load embeddings from each expert
if all_embeddings_exist:
    print("Loading pre-computed embeddings...")

    # Load mpnet embeddings
    mpnet_data = np.load(mpnet_file)
    train_mpnet = mpnet_data["train_embeddings"]
    val_mpnet = mpnet_data["val_embeddings"]
    test_mpnet = mpnet_data["test_embeddings"]

    # Load sentiment-specific embeddings
    sentiment_data = np.load(sentiment_file)
    train_sentiment = sentiment_data["train_embeddings"]
    val_sentiment = sentiment_data["val_embeddings"]
    test_sentiment = sentiment_data["test_embeddings"]

    # Load enhanced features
    enhanced_data = np.load(enhanced_file)
    train_features = enhanced_data["train_features"]
    val_features = enhanced_data["val_features"]
    test_features = enhanced_data["test_features"]

    print(
        f"Loaded mpnet embeddings: Train {train_mpnet.shape}, Val {val_mpnet.shape}, Test {test_mpnet.shape}"
    )
    print(
        f"Loaded sentiment embeddings: Train {train_sentiment.shape}, Val {val_sentiment.shape}, Test {test_sentiment.shape}"
    )
    print(
        f"Loaded enhanced features: Train {train_features.shape}, Val {val_features.shape}, Test {test_features.shape}"
    )

else:
    print("Generating embeddings from multiple experts...")

    # Expert 1: General purpose embeddings (mpnet)
    print("Loading mpnet model...")
    mpnet_model = SentenceTransformer("all-mpnet-base-v2")

    print("Generating mpnet embeddings...")
    start_time = time.time()
    train_mpnet = mpnet_model.encode(
        train_sentences, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    val_mpnet = mpnet_model.encode(
        val_sentences, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    test_mpnet = mpnet_model.encode(
        test_data["sentence"].values, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    print(f"mpnet embeddings generated in {time.time() - start_time:.2f} seconds")

    # Expert 2: Sentiment-specific embeddings
    print("Loading sentiment-specific model...")
    sentiment_model = SentenceTransformer(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Generating sentiment embeddings...")
    start_time = time.time()
    train_sentiment = sentiment_model.encode(
        train_sentences, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    val_sentiment = sentiment_model.encode(
        val_sentences, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    test_sentiment = sentiment_model.encode(
        test_data["sentence"].values, show_progress_bar=True, batch_size=BATCH_SIZE
    )
    print(f"Sentiment embeddings generated in {time.time() - start_time:.2f} seconds")

    # Expert 3: Explicit sentiment features
    print("Extracting sentiment-specific features...")
    start_time = time.time()
    train_features = extract_sentiment_features(train_sentences)
    val_features = extract_sentiment_features(val_sentences)
    test_features = extract_sentiment_features(test_data["sentence"].values)
    print(f"Sentiment features extracted in {time.time() - start_time:.2f} seconds")

    # Save all embeddings
    print("Saving embeddings...")
    np.savez_compressed(
        mpnet_file,
        train_embeddings=train_mpnet,
        val_embeddings=val_mpnet,
        test_embeddings=test_mpnet,
    )

    np.savez_compressed(
        sentiment_file,
        train_embeddings=train_sentiment,
        val_embeddings=val_sentiment,
        test_embeddings=test_sentiment,
    )

    np.savez_compressed(
        enhanced_file,
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
    )

    print("All embeddings saved successfully.")


# 5. Define the Mixture of Experts model
class SentimentMoE(nn.Module):
    def __init__(self, mpnet_dim=768, sentiment_dim=768, feature_dim=8):
        super(SentimentMoE, self).__init__()

        # Dimensions for each expert
        self.mpnet_dim = mpnet_dim
        self.sentiment_dim = sentiment_dim
        self.feature_dim = feature_dim

        # Expert-specific processing layers
        self.mpnet_processor = nn.Sequential(
            nn.Linear(mpnet_dim, 256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.sentiment_processor = nn.Sequential(
            nn.Linear(sentiment_dim, 256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(), nn.Dropout(0.2)
        )

        # Gating network to determine the importance of each expert
        self.gate = nn.Sequential(
            nn.Linear(mpnet_dim + sentiment_dim + feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),  # 3 experts
            nn.Softmax(dim=1),
        )

        # Fusion layer to combine expert outputs
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 3),  # 3 sentiment classes
        )

    def forward(self, mpnet_emb, sentiment_emb, features):
        # Process each expert's embeddings
        mpnet_out = self.mpnet_processor(mpnet_emb)
        sentiment_out = self.sentiment_processor(sentiment_emb)
        feature_out = self.feature_processor(features)

        # Calculate expert weights using the gate
        combined_input = torch.cat([mpnet_emb, sentiment_emb, features], dim=1)
        expert_weights = self.gate(combined_input)

        # Weighted combination of expert outputs (with residual connections)
        mpnet_weighted = mpnet_out * expert_weights[:, 0].unsqueeze(1)
        sentiment_weighted = sentiment_out * expert_weights[:, 1].unsqueeze(1)
        feature_weighted = feature_out * expert_weights[:, 2].unsqueeze(1)

        # Concatenate the weighted expert outputs
        combined = torch.cat(
            [mpnet_weighted, sentiment_weighted, feature_weighted], dim=1
        )

        # Final classification
        logits = self.fusion(combined)
        return logits, expert_weights


# 6. Setup training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Adjust labels for PyTorch CrossEntropyLoss
train_labels_adjusted = np.array([l + 1 for l in train_labels])
val_labels_adjusted = np.array([l + 1 for l in val_labels])

# Create the model
model = SentimentMoE(
    mpnet_dim=train_mpnet.shape[1],
    sentiment_dim=train_sentiment.shape[1],
    feature_dim=train_features.shape[1],
).to(device)

# Create dataset using all three types of embeddings
train_dataset = TensorDataset(
    torch.FloatTensor(train_mpnet),
    torch.FloatTensor(train_sentiment),
    torch.FloatTensor(train_features),
    torch.LongTensor(train_labels_adjusted),
)

val_dataset = TensorDataset(
    torch.FloatTensor(val_mpnet),
    torch.FloatTensor(val_sentiment),
    torch.FloatTensor(val_features),
    torch.LongTensor(val_labels_adjusted),
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)

# Class weights for imbalanced data
class_counts = np.bincount(train_labels_adjusted)
class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

# 7. Training loop
best_val_score = 0.0
patience_counter = 0

print("\nStarting training:")
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    # Training phase
    model.train()
    train_loss = 0.0
    expert_weights_sum = torch.zeros(3).to(device)
    samples_count = 0

    for mpnet_emb, sentiment_emb, features, labels in train_loader:
        # Move data to device
        mpnet_emb = mpnet_emb.to(device)
        sentiment_emb = sentiment_emb.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, expert_weights = model(mpnet_emb, sentiment_emb, features)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Tracking
        train_loss += loss.item()
        expert_weights_sum += expert_weights.sum(dim=0).detach()
        samples_count += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_expert_weights = expert_weights_sum / samples_count

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    val_expert_weights_sum = torch.zeros(3).to(device)
    val_samples_count = 0

    with torch.no_grad():
        for mpnet_emb, sentiment_emb, features, labels in val_loader:
            # Move data to device
            mpnet_emb = mpnet_emb.to(device)
            sentiment_emb = sentiment_emb.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, expert_weights = model(mpnet_emb, sentiment_emb, features)
            loss = criterion(logits, labels)

            # Predictions
            _, predicted = torch.max(logits, 1)

            # Tracking
            val_loss += loss.item()
            all_preds.extend((predicted.cpu().numpy() - 1).tolist())
            all_labels.extend((labels.cpu().numpy() - 1).tolist())
            val_expert_weights_sum += expert_weights.sum(dim=0).detach()
            val_samples_count += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_expert_weights = val_expert_weights_sum / val_samples_count

    # Calculate validation score
    mae_val = mean_absolute_error(all_labels, all_preds)
    val_score = 0.5 * (2 - mae_val)

    # Update learning rate based on validation score
    scheduler.step(val_score)

    # Display results
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[-1, 0, 1])
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Time: {epoch_time:.2f}s")
    print(
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Score: {val_score:.4f}"
    )
    print(
        f"Train Expert Weights: MPNet {avg_expert_weights[0]:.3f}, Sentiment {avg_expert_weights[1]:.3f}, Features {avg_expert_weights[2]:.3f}"
    )
    print(
        f"Val Expert Weights: MPNet {avg_val_expert_weights[0]:.3f}, Sentiment {avg_val_expert_weights[1]:.3f}, Features {avg_val_expert_weights[2]:.3f}"
    )
    print("Confusion Matrix:")
    print(conf_matrix)

    # Early stopping and model saving
    if val_score > best_val_score:
        best_val_score = val_score
        patience_counter = 0

        # Save model state dict
        torch.save(model.state_dict(), "best_sentiment_moe_model.pt")

        # Save metadata
        with open("best_model_metadata.txt", "w") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(f"val_score: {best_val_score}\n")
            f.write(f"mpnet_weight: {avg_val_expert_weights[0].item()}\n")
            f.write(f"sentiment_weight: {avg_val_expert_weights[1].item()}\n")
            f.write(f"features_weight: {avg_val_expert_weights[2].item()}\n")

        print(f"Model saved with score: {best_val_score:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# 8. Load best model for inference
try:
    model.load_state_dict(torch.load("best_sentiment_moe_model.pt"))

    # Read metadata if available
    try:
        with open("best_model_metadata.txt", "r") as f:
            metadata = {}
            for line in f:
                key, value = line.strip().split(": ")
                if key in [
                    "val_score",
                    "mpnet_weight",
                    "sentiment_weight",
                    "features_weight",
                ]:
                    value = float(value)
                metadata[key] = value
        print(
            f"Loaded best model from epoch {metadata.get('epoch', 'unknown')} with validation score {metadata.get('val_score', 'unknown')}"
        )
        print(
            f"Expert weights: MPNet {metadata.get('mpnet_weight', '?'):.3f}, Sentiment {metadata.get('sentiment_weight', '?'):.3f}, Features {metadata.get('features_weight', '?'):.3f}"
        )
    except:
        print("Loaded best model (metadata not available)")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using current model state without loading checkpoint")

# 9. Generate predictions on test data
model.eval()
test_preds = []
test_expert_weights_sum = torch.zeros(3).to(device)
test_samples_count = 0

print("Generating predictions for test data...")
start_time = time.time()

# Process test data in batches
for i in range(0, len(test_mpnet), TEST_BATCH_SIZE):
    batch_mpnet = torch.FloatTensor(test_mpnet[i : i + TEST_BATCH_SIZE]).to(device)
    batch_sentiment = torch.FloatTensor(test_sentiment[i : i + TEST_BATCH_SIZE]).to(
        device
    )
    batch_features = torch.FloatTensor(test_features[i : i + TEST_BATCH_SIZE]).to(
        device
    )

    with torch.no_grad():
        logits, expert_weights = model(batch_mpnet, batch_sentiment, batch_features)
        _, predicted = torch.max(logits, 1)
        test_preds.extend((predicted.cpu().numpy() - 1).tolist())

        # Track expert weights
        test_expert_weights_sum += expert_weights.sum(dim=0).detach()
        test_samples_count += batch_mpnet.size(0)

# Calculate average expert weights for test data
avg_test_expert_weights = test_expert_weights_sum / test_samples_count

print(f"Test predictions completed in {time.time() - start_time:.2f} seconds")
print(
    f"Test Expert Weights: MPNet {avg_test_expert_weights[0]:.3f}, Sentiment {avg_test_expert_weights[1]:.3f}, Features {avg_test_expert_weights[2]:.3f}"
)

# 10. Create submission file
submission = pd.DataFrame(
    {"id": test_data.index, "label": [reverse_mapping[pred] for pred in test_preds]}
)
submission.to_csv("submission.csv", index=False)
print("Submission file created!")

# 11. Print summary information
print("\nModel Performance Summary:")
print(f"Best validation score: {best_val_score:.4f}")
print("\nLabel distribution in training set:")
for label, count in zip(["Negative", "Neutral", "Positive"], class_counts):
    print(f"{label}: {count} ({count / sum(class_counts) * 100:.2f}%)")
