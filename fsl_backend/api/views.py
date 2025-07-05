import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand]).flatten()

def process_video(video_path, sequence_length=30):
    sequence = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sequence_length)

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        frame_count = 0
        while cap.isOpened() and len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            frame_count += 1

    cap.release()
    while len(sequence) < sequence_length:
        sequence.append(np.zeros(225))
    return np.array(sequence)

@csrf_exempt
def predict_sign(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        temp_video_path = f'media/{video_file.name}'
        os.makedirs('media', exist_ok=True)
        with open(temp_video_path, 'wb+') as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        sequence = process_video(temp_video_path)
        input_data = np.expand_dims(sequence.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction_idx = int(np.argmax(output_data))
        confidence = float(np.max(output_data))

        labels = ['hello', 'ikinagagalak kong makilala ka', 'magkita tayo bukas']

        os.remove(temp_video_path)

        return JsonResponse({'prediction': labels[prediction_idx], 'confidence': confidence})
    return JsonResponse({'error': 'Invalid request'}, status=400)











