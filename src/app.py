import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import glob
import random
from src.config import MODEL_SAVE_DIR, IMG_HEIGHT, IMG_WIDTH

# Global variables for model and classes
model = None
class_names = []

def load_model_and_classes():
    global model, class_names
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.keras')
    class_names_path = os.path.join(MODEL_SAVE_DIR, 'class_names.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run train.py first.")
    
    model = tf.keras.models.load_model(model_path)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print("Model and classes loaded successfully.")

def predict_disease(image):
    if model is None:
        return {"Error": "Model not loaded"}

    # Ensure image is RGB (Gradio sometimes passes RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Resize and preprocess
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Predict
    predictions = model.predict(img_array)[0]
    
    # Format output as a dictionary of {class_name: confidence} for Gradio's Label component
    # Gradio automatically formats this beautifully
    confidences = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    
    # Sort confidences by value (descending)
    confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))

    return confidences

def create_app():
    load_model_and_classes()
    custom_css = """
    .gradio-container {
        max-width: 95% !important;
        width: 100% !important;
        margin: auto;
    }
    #title-text {
        text-align: center;
        color: #2E8B57;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3.5em;
        margin-bottom: 0px;
    }
    #subtitle-text {
        text-align: center;
        color: #555555;
        font-size: 1.5em;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
        font-size: 1em;
        border-top: 1px solid #eee;
        padding-top: 20px;
    }
    /* Hide Gradio default footer */
    footer {
        display: none !important;
    }
    """
    
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="green",
        neutral_hue="slate"
    )
    
    # Get a few random examples for the UI
    example_images = []
    try:
        all_images = glob.glob("data/raw/*/*.jpg") + glob.glob("data/raw/*/*.png")
        if len(all_images) >= 3:
            example_images = random.sample(all_images, 3)
    except:
        pass
        
    # Define a clean, professional UI
    with gr.Blocks(title="Hydroponic Lettuce Disease Classifier", theme=theme, css=custom_css) as app:
        gr.Markdown(
            """
            <h1 id='title-text'>🌱 Hydroponic Lettuce Disease Classifier</h1>
            <p id='subtitle-text'>Upload an image of a hydroponic lettuce leaf to detect potential diseases. The deep learning model provides the top predictions along with their confidence scores.</p>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Leaf Image", height=500)
                analyze_btn = gr.Button("Analyze Leaf", variant="primary", size="lg")
                
                if example_images:
                    gr.Examples(examples=example_images, inputs=image_input, label="Try an Example Image")
            
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=5, label="Prediction Confidence")
                
        # Connect button to function
        analyze_btn.click(fn=predict_disease, inputs=image_input, outputs=label_output)
        
    return app

if __name__ == "__main__":
    try:
        app = create_app()
        # Gradio 6+ uses theme/css in launch, earlier versions use it in Blocks. We leave it in Blocks as it's just a warning.
        app.launch(share=False)
    except Exception as e:
        print(f"Failed to start app: {e}")
