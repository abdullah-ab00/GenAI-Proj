import os
import random
from flask import Flask, request, render_template
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Initialize Flask app
app = Flask(__name__)

# Load model directly
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to introduce noise in the caption by adding two words
def introduce_noise(caption, noise_level=0.2):
    words = caption.split()  # Split caption into words
    
    # Expanded list of random noise words
    random_words = random.sample(
        ['dog', 'cat', 'tree', 'car', 'bicycle', 'sky', 'mountain', 'cloud', 'apple', 'ball', 'bird', 'fish', 'house', 
         'grass', 'river', 'computer', 'table', 'phone', 'plane', 'city'], 
        2  # Picking 2 random words
    )
    
    # Insert one word in the middle
    middle_index = len(words) // 2  # Middle position
    words = words[:middle_index] + [random_words[0]] + words[middle_index:]  # Insert the first word in the middle
    
    # Randomly choose position (start or end) for the second word
    position = random.choice(['start', 'end'])  # Choose where to insert the second word
    
    if position == 'start':
        words = random_words[1:] + words  # Insert at the start
    else:
        words = words + random_words[1:]  # Insert at the end
    
    # Join words back into a string and return the "noisy" caption
    noisy_caption = " ".join(words)
    return noisy_caption

# Caption generation function using the Hugging Face model
def generate_caption(model, processor, img):
    # Preprocess the image and prepare the input
    inputs = processor(images=img, return_tensors="pt")
    
    # Generate the caption from the image
    out = model.generate(**inputs)
    
    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Introduce noise into the caption (insert two random noise words)
    noisy_caption = introduce_noise(caption, noise_level=0.2)  # Add two random noise words
    
    return noisy_caption

# Flask Routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")  # Open image and convert to RGB

    # Generate the caption using the pretrained model
    caption = generate_caption(model, processor, img)

    return render_template('result.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
