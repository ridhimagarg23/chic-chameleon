from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from backend.skin_color_detection2 import get_skin_color
from bg_remover import remove_background
from gender_detection import gender_detector
from bodyshape_detector import predict_body_shape

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './assets'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo provided"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Rename all uploaded files to "image" with their original extension
    extension = file.filename.rsplit('.', 1)[-1].lower()
    filename = f"image.{extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the file (overwrite if exists)
    file.save(file_path)

    image = file_path

    # AI analysis
    image = remove_background(image)
    skin_tone = get_skin_color(image)
    gender = gender_detector(image)
    body_shape = predict_body_shape(image, gender, waist, hip, bust)
    return jsonify({"gender": gender})


@app.route('/final_analyze', methods=['GET'])
def final_analyze():
    # Mock final analysis results
    results = {
        "photo_url": "./backend/static/image.png",
        "gender": "female",
        "skin_tone": "Fair",
        "skin_color": "#f8d9c0",
        "body_shape": "Hourglass"
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

# from g4f.client import Client

# client = Client()
# response = client.chat.completions.create(
#     model="gpt-4",  # ya koi available model
#     messages=[{"role": "user", "content": """
#     Based on the following user details:
#     - Gender: female
#     - Body Type: pear
#     - Skin Tone: fair
#      Skin Color -  #d5ab81
#      Event Type - Formal


#     Please provide:
#     1. A list or dictionary of **colors that suit** the user's skin tone along with their corresponding color codes (e.g., '#FF5733').
#     2. A list or dictionary of **dress types** that are recommended based on the EVENT TYPE AND  user's body type ANSWER BASED ON EVENT TYPE ONLY.
#     3. A list of **at least 10 do's** related to the body type and skin tone.
#     4. A list of **at least 10 don'ts** related to the body type and skin tone.

#     Respond in a structured format as JSON:
#     {{
#         "colors_suited": [{{"color": "color_name", "code": "color_code"}}],
#         "dress_recommendations": ["dress_type1", "dress_type2", ...],
#         "dos": ["do1", "do2", ...],
#         "donts": ["dont1", "dont2", ...]
#     }}
#     """}]
# )

# print(response.choices[0].message.content)
