import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from backend.bodyshape_detector import manual_body_shape_classification, body_type
# from bg_remover import remove_background
# from gender_detection import gender_detector
# from skin_color_detection import get_skin_color
# from outfit_gen import generate_all_colors
# from bodyshape_detector import classify_body_type
# import gender_detection
# import skin_color_detection

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = '../frontend/assets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the initial HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload and display the measurements form
@app.route('/submit', methods=['POST'])
def submit_image():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    event = request.form.get('event')

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Process image with bgremover
        # remove_background(image_path)
        #
        # # Detect gender automatically from the image
        # gender = gender_detector(image_path)
        #
        # # Detect skin color from the image
        # skin_color = get_skin_color(image_path)

        # Render the page with the image, event type, gender, skin color, and show the measurements form
        return render_template('../frontend/index.html',
                               image_name=filename,
                               gender=gender,
                               skin_color=skin_color,
                               event=event,
                               show_measurements=True)


# Route to process the measurements and generate outfit suggestions
@app.route('/process', methods=['POST'])
def process_measurements():
    # Get form data
    gender = request.form.get('gender')
    skin_color = request.form.get('skin_color')
    event_type = request.form.get('event')
    chest = float(request.form.get('chest'))
    waist = float(request.form.get('waist'))
    hips = float(request.form.get('hips'))

    # Run body shape analysis
    body_type = classify_body_type(gender, chest, waist, hips)

    # Run outfit generator
    outfits = generate_all_colors(event_type, body_type, gender, skin_color)

    # Render the outfits with the results
    return render_template('index.html',
                           outfits=outfits,
                           show_results=True,
                           event=event_type)


if __name__ == '__main__':
    app.run(debug=True)
