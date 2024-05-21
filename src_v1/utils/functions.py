import numpy as np 
import mediapipe as mp 
import os
import cv2
import keyboard
import tensorflow as tf

def make_folders(actions, no_sequences, DATA_PATH):

    '''function which makes as many folders as needed for data collection
        and saves them in the requires path'''

    for action in actions: 
        for frame_num in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(frame_num)))
            except:
                pass

def extract_keypoints(results, width, height):

    '''function which extracts the keypoint values from mediapipe hands results
        normalizes values using width and height to give back relative positions
        21 keypoints with x,y,z values'''


    keypoint_values = []
    hand_no = 0

    if results.multi_hand_landmarks:
        for hand_no, handLandmarks in enumerate(results.multi_hand_landmarks):
            hand_no =+ 1
            for point in mp.solutions.hands.HandLandmark:
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    normalizedLandmark.x, normalizedLandmark.y, width, height)
                keypoint_values.append([normalizedLandmark.x, normalizedLandmark.y, handLandmarks.landmark[point].z])
                keypoint_array = np.array(keypoint_values)
                keypoint_array_flat = keypoint_array.flatten()
        
        if (hand_no == 1) and (len(keypoint_array_flat) < 126): 
            zero_array = np.zeros(63)
            if results.multi_handedness[0].classification[0].label == 'Right':
                keypoint_array_flat = np.append(keypoint_array_flat, zero_array)
            elif results.multi_handedness[0].classification[0].label == 'Left':
                keypoint_array_flat = np.append(zero_array, keypoint_array_flat)

        return (keypoint_array_flat)


def import_solutions():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return(mp_hands, mp_drawing, mp_drawing_styles)


def frame_colection(key_action, key_to_press, count, input=0):
    
    mp_hands, mp_drawing, mp_drawing_styles = import_solutions()

    cap = cv2.VideoCapture(input)
    with mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.7) as hands:

        while cap.isOpened():

            success, image = cap.read()

            if not success:
                cap.release()
                cv2.destroyAllWindows()
            else:

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                image.flags.writeable = False
                image = image
                cv2.imshow('frame', cv2.flip(image, 1))
                results = hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

                frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                sequence_length = 20
                DATA_PATH = os.path.join('./data/processed_data/MP_Data') 

                if keyboard.is_pressed(str(key_to_press)):
                    print('keyboard pressed: {}'.format(key_to_press))
                    for frame_num in range(sequence_length):
                        results = hands.process(image)
                        keypoints = extract_keypoints(results, frameWidth, frameHeight)
                        action = key_action[key_to_press]
                        npy_path = os.path.join(DATA_PATH, action, str(count), str(frame_num))
                        print(os.path.join(DATA_PATH, action, str(count), str(frame_num)))
                        cv2.imwrite(DATA_PATH+str(frame_num)+'.jpg', image)
                        np.save(npy_path, keypoints)
                        success, image = cap.read()
                        frame_num = frame_num +1
                        image.flags.writeable = False
                        image = image
                        cv2.imshow('frame', cv2.flip(image, 1))
                        results = hands.process(image)
                        image.flags.writeable = True

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    image,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
                        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        cap.release()
        cv2.destroyAllWindows()


def make_variables(DATA_PATH, no_sequences, labels_dict, actions, sequence_length=10):

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
                window.append(res)
            sequences.append(window)
            labels.append(labels_dict[action])

    return(sequences, labels)

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)

    return(model)