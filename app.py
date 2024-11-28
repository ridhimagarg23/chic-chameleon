import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import your function files
# from gender_detection import gender_detector
# from outfit_gen import generate_all_colors


app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = './assets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route to serve the initial HTML page (index.html) on '/' route
@app.route('/')
def index():
    # Just render the index page first
    return render_template('index.html')


# Route to handle the image upload and processing
@app.route('/submit', methods=['POST'])
def submit_image():
    try:
        # Step 1: Validate if an image was uploaded
        if 'image' not in request.files:
            return "No file part", 400

        image = request.files['image']
        event = request.form.get('event')

        # Step 2: Validate and save the image
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            # Now that image is saved, process it
            from bg_remover import remove_background
            remove_background(image_path,"./assets/image.png")
            from gender_detection import gender_detector
            gender = gender_detector(image_path)
            from skin_color_detection import skin_color_code
            skin_color = skin_color_code

            # Step 3: Render the page with processed results
            return render_template(
                'index.html',
                image_name=filename,
                gender=gender,
                skin_color=skin_color,
                event=event,
                show_measurements=True,
            )
        else:
            return "Invalid file type", 400

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during processing. Please try again.", 500


# Route to process measurements and generate outfit suggestions
@app.route('/process', methods=['POST'])
def process_measurements():
    try:
        # Get form data
        gender = request.form.get('gender')
        skin_color = request.form.get('skin_color')
        event_type = request.form.get('event')
        chest = float(request.form.get('chest'))
        waist = float(request.form.get('waist'))
        hips = float(request.form.get('hips'))

        # Run body shape analysis
        from bodyshape_detector import body_type
        body_shape = body_type

        # Run outfit generator
        from outfit_gen import generate_all_colors
        outfits = generate_all_colors(event_type, body_shape, gender, skin_color)


        # Render the outfits with results
        return render_template(
            'index.html',
            outfits=outfits,
            show_results=True,
            event=event_type,
        )
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing measurements.", 500


if __name__ == '__main__':
    app.run(debug=True)
