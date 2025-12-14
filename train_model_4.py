import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# --- ----------------------DATA AUGMENTATION-------------------
def augment_sequence(sequence, noise_factor=0.05, time_shift_max=5):
    augmented = []
    seq_array = np.array(sequence)

    # 1. Original
    augmented.append(seq_array)

    # 2. Ruido Gaussiano (Simula temblor de cámara/mano)
    noise = np.random.normal(0, noise_factor, seq_array.shape)
    augmented.append(seq_array + noise)

    # 3. Time Stretch (Simula velocidad variable: lento/rápido)
    if len(seq_array) > 5:
        indices = np.linspace(0, len(seq_array)-1, len(seq_array))
        random_warp = np.random.randint(-2, 3, len(indices)) 
        stretched_indices = np.clip(indices + random_warp, 0, len(seq_array)-1).astype(int)
        augmented.append(seq_array[stretched_indices])
    else:
        augmented.append(seq_array)

    # 4. Time Shift (Simula empezar la seña antes o después)
    shift = np.random.randint(-time_shift_max, time_shift_max+1)
    if shift != 0:
        if shift > 0: # Desplazar derecha
            shifted = np.vstack([np.repeat(seq_array[:1], shift, axis=0), seq_array[:-shift]])
        else: # Desplazar izquierda
            shifted = np.vstack([seq_array[-shift:], np.repeat(seq_array[-1:], abs(shift), axis=0)])
        augmented.append(shifted)
    else:
        augmented.append(seq_array)

    return augmented

# -----------------------------------CARGAR DATOS -------------------------------------------
DATA_PATH = os.path.join('MP_Data_LSM') 
actions = np.array(open("actions.txt").read().splitlines())
no_sequences = 120 # Dataset completo

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print(f"Cargando dataset ({len(actions)} clases)...")

first_seq_path = os.path.join(DATA_PATH, actions[0], '0')
print(first_seq_path)
detected_sequence_length = len(os.listdir(first_seq_path))
print(f"\nLongitud de secuencicña detectada: {detected_sequence_length} frames")

# Carga 
for action in actions:
    for sequence in range(no_sequences):
        window = []
        try:
            path = os.path.join(DATA_PATH, action, str(sequence))
            if not os.path.exists(path): continue
            
            # Orden estricto de lectura
            frame_files = sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))
            
            # Validación de integridad
            if len(frame_files) != detected_sequence_length: continue

            for frame_name in frame_files:
                res = np.load(os.path.join(path, frame_name))
                window.append(res)
            
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            pass

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# ----------------------------------------------- SPLIT -----------------------------
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.20,stratify=y, random_state=42)

print(f"Train Original: {X_train_raw.shape}")
print(f"Test (Intacto): {X_test.shape}")

# --- -------------------------------------DATA AUGMENTATION (Solo en Train) ---------------------
#x4
print("Aumentando datos...")
X_train_aug, y_train_aug = [], []

for i in range(len(X_train_raw)):
    lbl = y_train_raw[i]
    # Genera 4 versiones
    augs = augment_sequence(X_train_raw[i])
    for aug_seq in augs:
        X_train_aug.append(aug_seq)
        y_train_aug.append(lbl)

X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

print(f"Train Final: {X_train.shape}")

# --------------------------------------CONFIGURACION DEL MODELO --------------------------------------
model = Sequential()

# LSTM 1
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(detected_sequence_length, 258)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# LSTM 2
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Densa
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# Salida
model.add(Dense(actions.shape[0], activation='softmax'))

# --- -------------------------------------OPTIMIZADOR ------------------------------------
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#-----------------------------------------CALLBACKS--------------------------------------------------
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

checkpoint = ModelCheckpoint('lsm_ultimate_model.keras', monitor='val_categorical_accuracy', save_best_only=True, verbose=1)

#reducir learning rate cuando no mejore
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

#------------------------------------------ENTRENAMIENTO---------------------------------------------------
print("\nIniciando Entrenamiento...")
history = model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback, early_stopping, checkpoint, reduce_lr], validation_data=(X_test, y_test), batch_size=32)

# ------------------------------------------REPORTES -------------------------------------------------------
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
final_accuracy = accuracy_score(y_true, y_pred_classes)
print(f"\nAccuracy Final: {final_accuracy*100:.2f}%")


#Gráfica de Loss y Accuracy
def save_training_history(history, filename='curvas_entrenamiento.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    #Gráfica Accuracy
    ax1.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
    ax1.set_title('Precisión del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Gráfica Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Pérdida (Error)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename) 
    print(f"Gráfica guardada")
    #plt.show()


#Matriz de confusión
def save_confusion_matrix(model, X_test, y_test, actions, filename='matriz_confusion.png'):
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadera Etiqueta')
    plt.xlabel('Predicción del Modelo')
    
    plt.savefig(filename)
    print(f"Matriz guardada")
    #plt.show()
    
    return accuracy_score(y_true, y_pred_classes)

#Guardar Gráficas
save_training_history(history)

#Guardar Matriz y calcular Score Final
cm = save_confusion_matrix(model, X_test, y_test, actions)

#Reporte detallado
report = classification_report(y_true, y_pred_classes, target_names=actions)

with open('reporte_resultados.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("REPORTE DE ENTRENAMIENTO LSM\n")
    f.write("="*60 + "\n\n")
    
    f.write("CONFIGURACIÓN DEL MODELO:\n")
    f.write(f"  - Arquitectura: LSTM (3 capas: 64→128→64)\n")
    f.write(f"  - Data Augmentation: 4x (original + ruido + stretch + shift)\n\n")
    
    f.write("DATASET:\n")
    f.write(f"  - Clases: {len(actions)} palabras\n")
    f.write(f"  - Secuencias originales por clase: {no_sequences}\n")
    f.write(f"  - Total muestras (Train + Test): {X_train.shape[0] + X_test.shape[0]}\n") # CORREGIDO
    f.write(f"  - Longitud de secuencia: {detected_sequence_length} frames\n")
    f.write(f"  - Dimensión de features: 258 (MediaPipe landmarks)\n\n")
    
    f.write("DATOS DE ENTRENAMIENTO:\n")
    f.write(f"  - Train: {X_train.shape[0]} secuencias\n")
    f.write(f"  - Test: {X_test.shape[0]} secuencias (20%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("RESULTADO FINAL\n")
    f.write("="*60 + "\n")
    f.write(f"Accuracy Global: {final_accuracy*100:.2f}%\n\n")
    
    f.write("REPORTE POR CLASE:\n")
    f.write("-"*60 + "\n")
    f.write(report)
    f.write("\n")

print(f"\nReporte completo generado")