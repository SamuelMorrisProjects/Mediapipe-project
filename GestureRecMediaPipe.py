import mediapipe as mp 
import cv2  
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
print("Finished importing dependencies!\n\nThis was developed by Samuel Morris and is a demo of ML powered hand recongtion using Mediapipe and google's gesture model recognizer\n\nThe code can be made available upon request, if you have any questions feel free to email me at samuelmorris333221@gmail.com")

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)


print("This program requires the google hand gesture recognizer to function or a similar equivalent.\nThe hand gesture recognizer can be dowloaded at this url: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task\n\nWorks best in strong light behind a defined background.\n\n")
model_path = input("Input the recognizer model path (Do not include quotes): ")

base_options = BaseOptions(model_asset_path=model_path)
gesture_confidence = "1"

def globalize_gesture_confidence(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int,):
        """Intializes a global variable to be printed on Opencv2 screen\n
            Var is a tuple containing two nested lists it can be accessed so\n
            Ex. (gesture_confidence)[0][0][0] (This is the name of the gesture)
            """
        global gesture_confidence 
        gesture_confidence = ([[category.category_name for category in gesture] for gesture in result.gestures],[[category.score for category in gesture] for gesture in result.gestures])

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=globalize_gesture_confidence)


def dectection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 95, 150), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(103, 231, 250), thickness=2, circle_radius=2))
                            

with GestureRecognizer.create_from_options(options) as recognizer:
    hol = mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5)
    time_stamp=0
    print("Press lowercase q to exit the program\n\nIf you have any questions using the software please read the attached user manual.")
    while cap.isOpened():
        ret, frame = cap.read()
        time_stamp+=1
        image, results = dectection(frame, hol)
        recognition_result = recognizer.recognize_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),time_stamp,)
        draw_landmarks(image, results)
        try:
            cv2.putText(image, f'{(gesture_confidence)[0][0][0]}    {(float(gesture_confidence[1][0][0])):.2f}', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except IndexError:
             pass
        cv2.imshow('Gesture Rec', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
