from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

save_directory = "./ClassificationModel"

# Load the saved tokenizer and model
tokenizer_fine_tuned_tf = DistilBertTokenizer.from_pretrained(save_directory)
model_fine_tuned_tf = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

# Test text for prediction
test_text = "BMW X5 with 10 Cylinders, price around 250000 EUR"

# Tokenize the input text
tokenized_inputs = tokenizer_fine_tuned_tf(test_text, truncation=True, padding=True, return_tensors='tf')

# Ensure the input tensors are created correctly
if 'input_ids' in tokenized_inputs and 'attention_mask' in tokenized_inputs:
    with tf.device('/CPU:0'):  # Use appropriate device if available (e.g., '/GPU:0')
        # Pass the tokenized inputs to the model
        output_tf = model_fine_tuned_tf(**tokenized_inputs)

    # Get the predicted class index
    prediction_value_tf = tf.argmax(output_tf.logits, axis=1).numpy()[0]
else:
    # Handle missing input tensors
    prediction_value_tf = None

print("Predicted category (TensorFlow):", prediction_value_tf)

if int(prediction_value_tf) == 0 :
    print("Predicted category: Cars")

if int(prediction_value_tf) == 1 :
    print("Predicted category: Others")

if int(prediction_value_tf) == 2 :
    print("Predicted category: Property")

