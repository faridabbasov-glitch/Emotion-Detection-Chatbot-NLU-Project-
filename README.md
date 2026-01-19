**Overview**

This project implements a Natural Language Understanding (NLU) model to classify emotions in text. Using a pretrained DistilBERT model, it is fine-tuned on the Emotion Dataset
 to recognize multiple emotions including sadness, joy, love, anger, fear, and surprise.

The project also demonstrates real-time integration with a chatbot-style interface, where user input is analyzed for emotion and the bot responds with human-like replies based on the predicted emotion.

**Features**

1)Fine-tuning of a pretrained DistilBERT model for multi-class emotion classification.

2)High performance on the test set: Accuracy: 0.9270, F1 Score: 0.9268.

3)Real-time chatbot interface for emotion detection and interactive responses.

4)Confidence scores for predictions to indicate model certainty.

5)Ready for integration into chatbots, text analysis tools, or other NLU applications.

6)Clear separation of training and inference for reproducibility.

**Model Training**

1)The model was trained using the Hugging Face Trainer API.

2)Dataset: Emotion Dataset with training, validation, and test splits.

3)Metrics used: Accuracy and weighted F1 score.

**Final performance on the test set:**

Accuracy: 0.9270

F1 Score: 0.9268

The trained model and tokenizer are saved in the trained_model/ directory for reuse.

**Chatbot Integration**

1)The project includes a command-line chatbot interface.

2)User input is analyzed for emotion in real-time.

3)Each predicted emotion triggers a pre-written human-like response.

4)Supports exit commands to stop the conversation.

5)Demonstrates the practical integration of a fine-tuned NLU model into an interactive system.

**Metrics and Evaluation**

Accuracy: Overall correctness of the model’s predictions.

Weighted F1 Score: Balanced per-class performance.

High test set performance demonstrates the model’s reliability for real-world text inputs.

Classification results are interpretable and actionable in chatbot responses.
