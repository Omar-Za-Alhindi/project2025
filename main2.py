import flask
import werkzeug
import os
import concurrent.futures
import main  # Make sure `main.py` exists and has the right processing code

# Initialize Flask application
app = flask.Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure uploads folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store result
result = ''


@app.route('/', methods=['POST'])
def handle_request():
    """Handles file upload and image processing."""
    # Check if 'image' key exists in the request
    if 'image' not in flask.request.files:
        return flask.jsonify({"error": "No image file provided"}), 400

    imagefile = flask.request.files['image']

    # Check if the file has a valid filename
    if imagefile.filename == '':
        return flask.jsonify({"error": "No selected file"}), 400

    # Secure the filename and save it
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagefile.save(filepath)
    print(f"✅ File received: {filename}")

    global result
    # Process the file asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(main.main, filepath)  # Make sure this is correct
        result, headers = future.result()

    # Return result from the processing function
    response = main.resultString(result, headers)
    print(response)  # Print the response for debugging
    return response


@app.route('/result/', methods=['GET'])
def result_request():
    """Handles result fetching."""
    if main.returnResult() != '':
        return main.returnResult()
    return 'Processing...'


# Optional route to handle favicon request
@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                                     mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
