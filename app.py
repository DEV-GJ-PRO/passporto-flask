from flask import Flask, render_template, request, send_file, Response
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from rembg import remove
import io
import os
import base64

app = Flask(__name__)

# Function to remove background and apply grey background
def remove_background(image, remove_bg=True):
    if not remove_bg:
        return image
    image_np = np.array(image)
    output = remove(image_np)
    output_img = Image.fromarray(output)
    grey_bg = Image.new('RGB', output_img.size, (255, 255, 255))
    if output_img.mode == 'RGBA':
        grey_bg.paste(output_img, (0, 0), output_img.split()[3])
    else:
        grey_bg.paste(output_img, (0, 0))
    return grey_bg.convert('RGB')

# Function to detect and center face
def detect_and_center_face(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center_x, face_center_y = x + w // 2, y + h // 2
        img_height, img_width = image_np.shape[:2]
        crop_size = int(max(w, h) * 1.5)
        crop_x1 = max(face_center_x - crop_size // 2, 0)
        crop_y1 = max(face_center_y - crop_size // 2, 0)
        crop_x2 = min(face_center_x + crop_size // 2, img_width)
        crop_y2 = min(face_center_y + crop_size // 2, img_height)
        cropped = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
        return Image.fromarray(cropped)
    return image

# Function to enhance image
def enhance_image(image):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    return image

# Function to resize image
def resize_image(image, target_width, target_height):
    image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    return image

# Function to compress image to target file size
def compress_to_size(image, target_size_kb):
    target_size = target_size_kb * 1024
    quality = 95
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    while quality > 10:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        size = buffer.tell()
        if size <= target_size:
            return buffer.getvalue()
        quality -= 5
    image = image.resize((int(image.width * 0.9), int(image.height * 0.9)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    return buffer.getvalue()

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        try:
            # Get uploaded file
            file = request.files.get('image')
            if not file:
                error = "No file uploaded"
                return render_template('index.html', error=error)

            # Load image
            image = Image.open(file).convert('RGB')

            # Get form data
            bg_option = request.form.get('bg_option', 'Keep Background')
            remove_bg = bg_option == 'Remove Background'
            target_width = int(request.form.get('target_width', 800))
            target_height = int(request.form.get('target_height', 600))
            target_size_kb = int(request.form.get('target_size_kb', 100))

            # Process image
            image = remove_background(image, remove_bg)
            image = detect_and_center_face(image)
            image = enhance_image(image)
            image = resize_image(image, target_width, target_height)
            processed_image_bytes = compress_to_size(image, target_size_kb)

            # Convert to base64 for display
            processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')

            return render_template('result.html', 
                                 processed_image=processed_image_base64,
                                 filename='processed_image.jpg')
        except Exception as e:
            error = f"Error processing image: {str(e)}"
            return render_template('index.html', error=error)

    return render_template('index.html', error=error)

# Download route
@app.route('/download/<filename>')
def download(filename):
    processed_image_bytes = request.args.get('image')
    if not processed_image_bytes:
        return "No image data", 400
    image_data = base64.b64decode(processed_image_bytes)
    return send_file(
        io.BytesIO(image_data),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=filename
    )

# Serve service worker
@app.route('/site/sw.js')
def serve_sw():
    return send_file('static/sw.js', mimetype='application/javascript')

# Privacy policy
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # default fallback port
    app.run(host="0.0.0.0", port=port)