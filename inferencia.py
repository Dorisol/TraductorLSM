import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

#Lista de palabras 
"""
actions = np.array(['adios','bien','buenas noches','buenas tardes','bueno','buenos dias',
                    'como estas','de nada','gracias','hola','mal','nada','no','no saber',
                    'nos vemos mañana','otra vez','por favor','quien es','si','si saber','ya','yo'])
"""
                  
actions = np.array(open("actions.txt").read().splitlines())


#----------------------------------CARGAR MODELO-------------------------------                    
sequence_length = 45         #cantidad de frames 
confianza = 0.95             #confianza mínima
ventana_estabilidad = 10       #ventana de estabilidad

print("Cargando modelo...")
model = load_model('si_funciona/prueba_3/lsm_best_model_dori.keras') 
#model = load_model('si_funciona/prueba_5/lsm_ultimate_model.keras') 
#model = load_model('lsm_ultimate_model.keras') 
#model = load_model('lsm_best_model.keras') 

print("Modelo cargado exitosamente!!")

#--------------------------------CARGAR MEDIAPIPE-----------------------------------------
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
    
    center_point = (left_shoulder + right_shoulder) / 2                #punto central
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)    #calcular ancho de hombros
    if shoulder_width < 0.001: shoulder_width = 1.0

    #print(f"Shoulder width: {shoulder_width}")

    #Normalizar manos
    def normalize_list(landmark_list, dims=3):
        #si no hay manos, retornar ceros
        if not landmark_list:
            return np.zeros(21*dims)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark])
        points = points - center_point     #centrar respecto al punto central
        points = points / shoulder_width   # Escalar respecto al ancho de hombros
        return points.flatten()

    #normalizar pose
    pose_raw = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
    pose_raw[:, :3] = (pose_raw[:, :3] - center_point) / shoulder_width
    pose = pose_raw.flatten()
    #normalizar manos
    lh = normalize_list(results.left_hand_landmarks)
    rh = normalize_list(results.right_hand_landmarks)
    return np.concatenate([pose, lh, rh])

#Mostrar top palabras
def prob_viz_top3(res, actions, input_frame):
    output_frame = input_frame.copy()
    
    # Indices de las 3 probabilidades más altas
    top_3_indices = np.argsort(res)[-3:][::-1]
    
    for i, idx in enumerate(top_3_indices):
        prob = res[idx]
        action_label = actions[idx]
        
        # Color: Verde si supera umbral, Amarillo si es sospechoso, Gris si es muy bajo
        if prob > confianza:
            color = (0, 255, 0) # Verde Puro
        elif prob > 0.5:
            color = (0, 255, 255) # Amarillo
        else:
            color = (100, 100, 100) # Gris
            
        # Barra de progreso
        cv2.rectangle(output_frame, (0, 60 + i*40), (int(prob * 250), 90 + i*40), color, -1)
        
        # Texto con sombra para legibilidad
        text = f"{action_label}: {prob*100:.1f}%"
        cv2.putText(output_frame, text, (5, 85 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA) # Sombra negra
        cv2.putText(output_frame, text, (5, 85 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # Texto blanco
        
    return output_frame


sequence = [] # Buffer de frames
current_word = ""
predictions = []  #historial de predicciones

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:

    print("Camara iniciada. Presiona 'q' para salir.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Detección
        image, results = mediapipe_detection(frame, holistic)
        
        # 2. Dibujar landmarks (Solo manos)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # 3.Predicción
        try:
            #Extraer keypoints y agregar al buffer
            keypoints = extract_keypoints_normalized(results)
            sequence.append(keypoints)
            
            # Mantener el buffer del tamaño exacto (45)
            sequence = sequence[-sequence_length:]
            
            #Predecir solo cuando haya 45 frames completos
            if len(sequence) == sequence_length:
                #Pasar secuencia al modeo
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                #Estabilizar (indice de mayor probabilidad) y guardar en historial
                best_class_idx = np.argmax(res)
                predictions.append(best_class_idx)
                
                # Revisar ventana de estabilidad
                if np.unique(predictions[-ventana_estabilidad:])[0] == best_class_idx: 
                    #Verificar umbral
                    if res[best_class_idx] > confianza: 
                        current_word = actions[best_class_idx]

                # Ver barras de probabilidad
                image = prob_viz_top3(res, actions, image)
                
        except Exception as e:
            print(f"Error en inferencia: {e}")
            pass
            
        # Mostrar palabra detectada
        cv2.rectangle(image, (0,0), (640, 40), (30, 30, 30), -1)
        cv2.putText(image, current_word, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Traductor LSM', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()