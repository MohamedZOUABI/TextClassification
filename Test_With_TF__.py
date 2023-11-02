from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

save_directory = "./ClassificationModel"

# Load the saved tokenizer and model
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

# Test text for prediction
# test_text = "i am looking for electric guitar"
# test_text = "is there any studio for rent in La Marsa"
# test_text = "BMW X5 with 10 Cylinders, price around 250000 EUR"

test_text = input("What are you looking for : ")


# Tokenize the test text
predict_input = tokenizer_fine_tuned.encode(
    test_text,
    truncation=True,
    padding=True,
    return_tensors='tf'
)

# Get model prediction
output = model_fine_tuned(predict_input)[0]
prediction_value = tf.argmax(output, axis=1).numpy()[0]

# Output the prediction
# print("Predicted category:", prediction_value)

# print("Text : "+ test_text)

if int(prediction_value) == 0 :
    print("Predicted category: Cars")

if int(prediction_value) == 1 :
    print("Predicted category: Others")

if int(prediction_value) == 2 :
    print("Predicted category: Property")



