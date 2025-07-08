import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import aiofiles
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
import asyncio

# Load the model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe setup
mp_holistic = mp.solutions.holistic

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand]).flatten()

# Process video: optimized
def process_video(video_path, sequence_length=30):
    sequence = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 5 if total_frames <= 60 else 10  # skip more frames if long

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        frame_count = 0
        while cap.isOpened() and len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                frame = cv2.resize(frame, (480, 360))  # faster processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            frame_count += 1

    cap.release()
    while len(sequence) < sequence_length:
        sequence.append(np.zeros(225))
    return np.array(sequence)

# Save file async to /tmp/
async def save_uploaded_file_async(video_file, path):
    async with aiofiles.open(path, 'wb') as out_file:
        for chunk in video_file.chunks():
            await out_file.write(chunk)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def predict_sign(request):
    video_file = request.FILES.get('video')
    if not video_file:
        return Response({'error': 'No video file provided'}, status=400)

    try:
        # Save to /tmp/ for performance and avoid media issues
        temp_video_path = os.path.join('/tmp', video_file.name)

        # Save using async function
        if isinstance(video_file, (InMemoryUploadedFile, TemporaryUploadedFile)):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(save_uploaded_file_async(video_file, temp_video_path))
        else:
            return Response({'error': 'Unsupported file type'}, status=400)

        # Process
        sequence = process_video(temp_video_path)
        input_data = np.expand_dims(sequence.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        labels = ['hello', 'ikinagagalak kong makilala ka', 'magkita tayo bukas']

        os.remove(temp_video_path)

        return Response({'prediction': labels[prediction_idx], 'confidence': confidence})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)
