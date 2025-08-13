```python
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("cat_dog_model.h5")

def classify_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat vs Dog Classifier",
    description="Upload an image to check if it's a Cat or Dog."
)

iface.launch()
