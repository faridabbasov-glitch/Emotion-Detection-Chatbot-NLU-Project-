MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "emotion"
OUTPUT_DIR = "./results"
TRAINED_MODEL_DIR = "./trained_model"
MAX_LENGTH = 128

TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 8,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "warmup_steps": 500,
    "save_total_limit": 2,
}


EMOTION_RESPONSES = {
    "sadness":  "I'm sorry you're feeling down. I hope things get better soon. 😔",
    "joy":      "That sounds wonderful! I'm glad you're feeling happy! 😊",
    "love":     "That's so sweet! Love is a beautiful thing! ❤️",
    "anger":    "I can understand your frustration. Take a deep breath. 😤",
    "fear":     "That sounds concerning. I hope everything works out for you. 🤗",
    "surprise": "Wow, that's quite unexpected! 😲",
}
