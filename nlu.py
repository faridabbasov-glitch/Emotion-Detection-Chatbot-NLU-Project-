!pip install -q transformers datasets accelerate evaluate scikit-learn

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content')

from config import MODEL_NAME, DATASET_NAME, TRAINED_MODEL_DIR, MAX_LENGTH, TRAINING_ARGS

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

dataset = load_dataset(DATASET_NAME)
label_names = dataset["train"].features["label"].names

print(f"Train      : {len(dataset['train']):,} samples")
print(f"Validation : {len(dataset['validation']):,} samples")
print(f"Test       : {len(dataset['test']):,} samples")
print(f"Labels     : {label_names}")

train_labels = dataset["train"]["label"]
counts = [train_labels.count(i) for i in range(6)]

plt.figure(figsize=(10, 4))
bars = plt.bar(label_names, counts,
               color=["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6","#1abc9c"])
plt.title("Class Distribution in Training Set", fontsize=14)
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 10, str(count), ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/class_distribution.png", dpi=150)
plt.show()
print("Saved: class_distribution.png")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=MAX_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Tokenization tamamlandı")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_names))
print("Model yükləndi")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy" : accuracy_score(labels, predictions),
        "f1"       : f1_score(labels, predictions, average="weighted"),
    }

training_args = TrainingArguments(**TRAINING_ARGS)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

print("Training başlayır...")
trainer.train()
print("Training tamamlandı")

results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test Accuracy : {results['eval_accuracy']:.4f}")
print(f"Test F1       : {results['eval_f1']:.4f}")

predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = tokenized_datasets["test"]["label"]

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_names))

history = trainer.state.log_history

val_loss, val_acc, epochs = [], [], []
for log in history:
    if "eval_loss" in log:
        val_loss.append(log["eval_loss"])
        val_acc.append(log["eval_accuracy"])
        epochs.append(int(log["epoch"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, val_loss, color="#e74c3c", marker="o", label="Validation Loss")
ax1.set_title("Validation Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs, val_acc, color="#2ecc71", marker="o", label="Validation Accuracy")
ax2.set_title("Validation Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/loss_accuracy_curve.png", dpi=150)
plt.show()
print("Saved: loss_accuracy_curve.png")

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

model.config.id2label = {i: name for i, name in enumerate(label_names)}
model.config.label2id = {name: i for i, name in enumerate(label_names)}
trainer.save_model(TRAINED_MODEL_DIR)
tokenizer.save_pretrained(TRAINED_MODEL_DIR)
print(f"Model saxlanıldı: {TRAINED_MODEL_DIR}")

from ipywidgets import widgets
from IPython.display import display

inf_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
inf_model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_DIR)
inf_model.eval()

EMOTION_RESPONSES = {
    "sadness"  : "I'm sorry you're feeling down. 😔",
    "joy"      : "That sounds wonderful! 😊",
    "love"     : "That's so sweet! ❤️",
    "anger"    : "I can understand your frustration. 😤",
    "fear"     : "That sounds concerning. 🤗",
    "surprise" : "Wow, that's quite unexpected! 😲",
}

def predict_emotion(text):
    inputs = inf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = inf_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    return label_names[idx], probs[idx].item()

text_input  = widgets.Text(placeholder="Type your message here...",
                           layout=widgets.Layout(width="500px"))
out         = widgets.Output()
send_button = widgets.Button(description="Send", button_style="primary")
exit_button = widgets.Button(description="Exit", button_style="danger")

def on_send(b):
    with out:
        text = text_input.value.strip()
        if not text:
            return
        emotion, confidence = predict_emotion(text)
        print(f"You     : {text}")
        print(f"Emotion : {emotion.upper()} ({confidence:.1%})")
        print(f"Bot     : {EMOTION_RESPONSES[emotion]}")
        print("-" * 50)
        text_input.value = ""

def on_exit(b):
    with out:
        print("Bot: Goodbye! 👋")
    text_input.disabled  = True
    send_button.disabled = True
    exit_button.disabled = True

send_button.on_click(on_send)
exit_button.on_click(on_exit)

print("=" * 50)
print("  EMOTION DETECTION CHATBOT")
print("=" * 50)
display(widgets.HBox([text_input, send_button, exit_button]), out)

