from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_FILE = Path(__file__).resolve().parent / "data" / "loan_data.csv"

df = pd.read_csv(DATA_FILE)

# convert categorical features via one-hot encoding
df = pd.get_dummies(
    df,
    columns=[
        'person_gender',
        'person_education',
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file'
    ]
)

# additional domain-inspired features
df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
df['credit_income_product'] = df['credit_score'] * df['person_income']
df['experience_income_ratio'] = (df['person_emp_exp'] + 1) / (df['person_income'] + 1)

# split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_status'])

# split the data into features and target
X_train = train_df.drop(columns=['loan_status'])
y_train = train_df['loan_status']

X_test = test_df.drop(columns=['loan_status'])
y_test = test_df['loan_status']

# ensure all features are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# compute class imbalance weight for positive class
class_counts = y_train.value_counts()
pos_weight_value = class_counts.get(0.0, 1.0) / class_counts.get(1.0, 1.0)

# scale features for stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert arrays to float32 numpy arrays
X_train_np = X_train.astype('float32', copy=False)
X_test_np = X_test.astype('float32', copy=False)
y_train_np = y_train.to_numpy(dtype='float32', copy=True)
y_test_np = y_test.to_numpy(dtype='float32', copy=True)

# convert numpy arrays to tensors
X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np).unsqueeze(1)

X_test_tensor = torch.from_numpy(X_test_np)
y_test_tensor = torch.from_numpy(y_test_np).unsqueeze(1)

full_train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100)

# create the model
input_dim = X_train_np.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 1)
)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# train the model with validation + early stopping
best_val_loss = float('inf')
patience = 15
patience_counter = 0
num_epochs = 500
best_state = model.state_dict()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_logits = model(X_val)
            loss = loss_fn(val_logits, y_val)
            val_loss += loss.item() * X_val.size(0)
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)


def find_best_threshold(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.sigmoid(model(X_batch))
            all_probs.append(probs)
            all_labels.append(y_batch)
    probs = torch.cat(all_probs).squeeze()
    labels = torch.cat(all_labels).squeeze()
    thresholds = torch.linspace(0.1, 0.9, steps=17)
    best_thresh = 0.5
    best_acc = 0.0
    for thresh in thresholds:
        preds = (probs >= thresh).float()
        acc = preds.eq(labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh.item()
    return best_thresh, best_acc


best_threshold, val_acc = find_best_threshold(model, val_loader)
print(f"Best validation accuracy {val_acc:.4f} at threshold {best_threshold:.2f}")


def accuracy(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    predicted = (probs >= threshold).float()
    return (predicted.eq(labels).float().mean().item())

with torch.no_grad():
    test_logits = model(X_test_tensor)
    loss = loss_fn(test_logits, y_test_tensor)
    print(f"Test Loss: {loss.item():.4f}")
    print(f"Test Accuracy: {accuracy(test_logits, y_test_tensor, threshold=best_threshold):.4f}")

model_path = Path(__file__).resolve().parent / "trained_model.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "input_dim": input_dim,
    "best_threshold": best_threshold,
    "pos_weight": pos_weight_value
}, model_path)
print(f"Model saved to {model_path}")