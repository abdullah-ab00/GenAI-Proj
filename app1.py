import json
import tensorflow as tf
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import os
from transformers import GPT2Tokenizer  # Importing the pretrained tokenizer
from tensorflow.python.keras.models import load_model

# Suppress TensorFlow INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Custom FeedForward Layer (as in your code)
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(rate=dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)
        return x

# Define the Captioner class with from_config method
class Captioner(tf.keras.Model):
    def __init__(self, feature_extractor, num_layers, num_heads, d_model, dff, pred_max_len=50, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = feature_extractor
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.pred_max_len = pred_max_len
        self.dropout_rate = dropout_rate

        # Define the custom layers
        self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=d_model)  # Adjust vocab size and embedding dimensions
        self.attention_layers = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model) for _ in range(num_layers)]
        self.ffn_layers = [FeedForward(d_model, dff) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(5000, activation='softmax')  # Vocabulary size and softmax activation

    def call(self, inputs, training=False):
        img_features, caption_input = inputs

        # Extract features from image
        x = self.feature_extractor(img_features)
        x = tf.expand_dims(x, axis=1)  # Add sequence dimension for the image feature

        # Process caption sequence with embeddings
        caption_embedded = self.embedding(caption_input)

        # Combine image features with caption input and process through attention layers and FFNs
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](caption_embedded, caption_embedded)  # Self-attention
            caption_embedded = self.ffn_layers[i](attention_output)  # Feedforward

        # Predict the next word (token) in the caption sequence
        output = self.output_layer(caption_embedded)
        return output

    @classmethod
    def from_config(cls, config):
        # Manually initialize hyperparameters if they're missing from the config
        num_layers = 6  # Default value for number of layers
        num_heads = 8   # Default value for number of attention heads
        d_model = 512   # Default value for model dimension
        dff = 2048      # Default value for feedforward network dimension
        pred_max_len = 50  # Max caption length
        dropout_rate = 0.1  # Default dropout rate

        # Initialize the feature extractor manually
        feature_extractor = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg',
            weights='models/feature_extractor.h5'  # Use ImageNet weights
        )
        feature_extractor.trainable = False  # Freeze the weights

        # Return an instance of Captioner with parameters from the config
        return cls(
            feature_extractor=feature_extractor,
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            dff=dff,
            pred_max_len=pred_max_len,
            dropout_rate=dropout_rate,
            **config  # Pass any additional config parameters to the parent class
        )

# Re-load your model from the h5 format
captioner_model = load_model('models/captioner_model.h5', custom_objects={'Captioner': Captioner})

# Save the model in TensorFlow SavedModel format
captioner_model.save('models/captioner_model_tf')

# Load the pretrained captioner model with custom_objects for Captioner class
captioner_model = load_model('models/captioner_model_tf', custom_objects={'Captioner': Captioner}, compile=False)

# Load the models for feature extractor and output layer
feature_extractor = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    pooling='avg',
    weights='models/feature_extractor.h5'  # Use ImageNet weights
)
feature_extractor.trainable = False  # Freeze the weights

# Caption generation function (using pretrained captioner_model)
def generate_caption(model, feature_extractor, tokenizer, img, max_len=50):
    img_features = feature_extractor(img)  # Extract features from image

    # Start token for the caption
    start_token = tokenizer.encode('[START]')[0]
    generated_caption = [start_token]

    for _ in range(max_len):  # Generate caption token by token
        predictions = model([img_features, tf.constant(generated_caption)], training=False)
        predicted_id = tf.argmax(predictions[:, -1, :], axis=-1, output_type=tf.int32)

        # Stop when [END] token is generated
        end_token = tokenizer.encode('[END]')[0]
        if predicted_id.numpy()[0] == end_token:
            break

        generated_caption.append(predicted_id.numpy())

    caption = tokenizer.decode(generated_caption[1:])  # Decode the generated tokens
    return caption

# Flask Routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    img = Image.open(file.stream).resize((224, 224))  # Resize image for feature extractor
    img_array = np.array(img) / 255.0  # Normalize the image
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    # Generate the caption using the pretrained model
    caption = generate_caption(captioner_model, feature_extractor, tokenizer, img_tensor)

    return render_template('result.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
