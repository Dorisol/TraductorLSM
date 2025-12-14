import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import warnings
warnings.filterwarnings("ignore")

# ------------------------ CONFIGURACION DEL PROYECTO ----------------------------
DATA_PATH = os.path.join('MP_Data_LSM_RUIDO') 
no_sequences = 30        #cant tomas    
sequence_length = 45     #tiempo 

# ------------------------ CONFIGURACIÓN DE MONITOREO -----------------------------
FPS_THRESHOLD = 30 
FRAME_LIMIT = 1.0 / FPS_THRESHOLD 

# ------------------------------- MEDIAPIPE ------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#Función de normalización de keypoints
def extract_keypoints_normalized(results):
    #Si no hay personas, retornar ceros
    if not results.pose_landmarks:
        return np.zeros(33*4 + 21*3 + 21*3)

    #landmarks de la pose y hombros
    landmarks = results.pose_landmarks.landmark
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])

    center_point = (left_shoulder + right_shoulder) / 2                 #punto central
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)     #calcular ancho de hombros
    if shoulder_width < 0.001: shoulder_width = 1.0

    #Normalizar manos
    def normalize_list(landmark_list, dims=3):
        #si no hay manos, retornar ceros
        if not landmark_list:
            return np.zeros(21*dims)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark])
        points = points - center_point      #centrar respecto al punto central
        points = points / shoulder_width    # Escalar respecto al ancho de hombros
        return points.flatten()

    # normalizar pose
    pose_raw = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
    pose_raw[:, :3] = (pose_raw[:, :3] - center_point) / shoulder_width
    pose = pose_raw.flatten()

    # normalizar manos
    lh = normalize_list(results.left_hand_landmarks)
    rh = normalize_list(results.right_hand_landmarks)
    
    return np.concatenate([pose, lh, rh])

def main():
    action = input("Ingresa el nombre de la seña a grabar: ").strip()

    # Solo creamos la carpeta base de la acción aquí
    if not os.path.exists(os.path.join(DATA_PATH, action)):
        os.makedirs(os.path.join(DATA_PATH, action))

    #Resolución baja 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #Optimizacion Modelo Litte
    with mp_holistic.Holistic(min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5,
                              model_complexity=0, # Lite para velocidad
                              smooth_landmarks=True) as holistic:
        
        # Esperar tecla ESPACIO antes de empezar todo el bloque
        print(f"Presiona ESPACIO para iniciar la grabación de '{action}'")
        while True:
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            
            # Dibujar para que el usuario se acomode
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.rectangle(image, (0,0), (640, 50), (245, 117, 16), -1)
            cv2.putText(image, f'Presiona ESPACIO para grabar: {action}', (15,35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)
            
            # Detectar ESPACIO
            if cv2.waitKey(10) & 0xFF == 32:
                break
            # Salir con 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # --------------------------- SECUENCIAS ---------------
        for sequence in range(no_sequences):
            
            # Pausa visual entre videos (Countdown)
            for i in range(40, 0, -1): 
                ret, frame = cap.read()
                cv2.putText(frame, f'PREPARATE: {action} - Video {sequence}', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Iniciando en {i}', (120,250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1)


            # BUFFER RAM
            window_buffer = [] 

            # --------------------- GRABACIÓN -------------------------
            for frame_num in range(sequence_length):
                start_time = time.time()

                #captura 
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)

                #Visualizacion
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Extraer y guardar 
                keypoints = extract_keypoints_normalized(results)
                window_buffer.append(keypoints)

                cv2.rectangle(image, (0,0), (640, 50), (0, 255, 0), -1)
                cv2.putText(image, f'Grabando: {action} [{sequence}] Frame: {frame_num}',(15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # Control de FPS manual
                processing_time = time.time() - start_time
                if processing_time > FRAME_LIMIT:
                    fps_real = 1.0 / processing_time
                    print(f"LAG: {fps_real:.1f} FPS")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # ----------------------- GUARDAR SECUENCIA------------------ ---
            print(f"Guardando secuencia {sequence}...")
            
            # 1. Definir ruta de LA CARPETA NUMÉRICA (ej: .../hola/0)
            folder_path = os.path.join(DATA_PATH, action, str(sequence))
            
            # 2. CREAR LA CARPETA SI NO EXISTE 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 3. Guardar los frames dentro de esa carpeta
            for frame_num, keypoints_data in enumerate(window_buffer):
                npy_path = os.path.join(folder_path, str(frame_num)) # Usamos folder_path directo
                np.save(npy_path, keypoints_data)

    cap.release()
    cv2.destroyAllWindows()


main()