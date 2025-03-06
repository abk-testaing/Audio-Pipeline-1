from flask import Flask, request, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict import explain_prediction
import os
import librosa
import json
from collections import OrderedDict
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from docs import create_api  # Import API documentation setup

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/')
def home():
    return Response(json.dumps({
        "status": {"type": "success", "message": "Audio Deepfake Detection API is running"}
    }), mimetype='application/json')

# ✅ Integrate Swagger API documentation
api = create_api(app)

# Initialize rate limiter with correct IP tracking
limiter = Limiter(
    app=app,
    key_func=lambda: request.headers.get("X-Forwarded-For", request.access_route[-1]),
    default_limits=[]
)

# Allowed file formats
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'aac'}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constraints
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB limit
MIN_AUDIO_DURATION = 3  # Min duration in seconds
MAX_AUDIO_DURATION = 30  # Max duration in seconds

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_audio_duration(file_path):
    try:
        duration = librosa.get_duration(path=file_path)
        if MIN_AUDIO_DURATION <= duration <= MAX_AUDIO_DURATION:
            return True, None
        return False, f"Audio duration must be between {MIN_AUDIO_DURATION} and {MAX_AUDIO_DURATION} seconds."
    except Exception as e:
        return False, str(e)

@app.before_request
def check_request_size():
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return Response(json.dumps({
            "status": {"type": "error", "message": "Request size exceeds 2MB limit"},
            "result": None
        }), mimetype='application/json', status=413)

@app.route('/predict', methods=['POST'])
@limiter.limit("15 per minute", key_func=lambda: request.headers.get("X-Forwarded-For", request.remote_addr))
def predict_audio():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return Response(json.dumps({
            "status": {"type": "error", "message": "Authentication Failed."},
            "result": None
        }), mimetype='application/json', status=401)

    token = auth_header.split(" ")[1]
    if token != "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY3OWY0ZjJlZGRlOTg0MTNlNWZlMGYxNyIsImlhdCI6MTczODUwNzIwNCwiZXhwIjoxNzM4NTEwODA0fQ.McTQ8C2dapo7DVklBe2HfiKHirx0P4KEjGJ15VYH0Qk":
        return Response(json.dumps({
            "status": {"type": "error", "message": "Authentication Failed."},
            "result": None
        }), mimetype='application/json', status=401)

    if 'file' not in request.files:
        return Response(json.dumps({
            "status": {"type": "error", "message": "No file uploaded"},
            "result": None
        }), mimetype='application/json', status=400)

    file = request.files['file']

    if file.filename == '':
        return Response(json.dumps({
            "status": {"type": "error", "message": "No file selected"},
            "result": None
        }), mimetype='application/json', status=400)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        is_valid_duration, error_message = validate_audio_duration(file_path)
        if not is_valid_duration:
            os.remove(file_path)
            return Response(json.dumps({
                "status": {"type": "error", "message": error_message},
                "result": None
            }), mimetype='application/json', status=400)

        try:
            prediction = explain_prediction(file_path)
            is_speechpresent = bool(prediction["status"])
            if is_speechpresent:
                probability = float(prediction["probability"])
                is_deepfake = bool(prediction["deepfake"])
                return Response(json.dumps({
                "status": {"type": "success", "message": "Prediction successful."},
                "result": {
                    "deepfake_prediction": is_deepfake,
                    "probability": probability
                }
            }), mimetype='application/json')
            else:
                return Response(json.dumps({
                "status":{"type": "error", "message": "No speech in audio, Provide an audio with speech."},
                "result":"None"
            }), mimetype='application/json', status=400)

        except Exception as e:
            return Response(json.dumps({
                "status": {"type": "error", "message": "Error during prediction", "details": str(e)}
            }), mimetype='application/json', status=500)

    return Response(json.dumps({
        "status": {"type": "error", "message": "Invalid file format. Only WAV, MP3, FLAC, AAC allowed."},
        "result": None
    }), mimetype='application/json', status=400)

@app.errorhandler(429)
def ratelimit_handler(error):
    return Response(json.dumps({
        "status": {"type": "error", "message": "Too many requests from this IP. Please try again later."},
        "result": None
    }), mimetype='application/json', status=429)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
