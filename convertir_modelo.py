import tensorflow as tf
import os
import shutil

def convert_lstm_model(input_path, output_path):
    temp_dir = "temp_saved_model_dir"
    
    #Limpieza inicial
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"1. Cargando modelo Keras 3 desde: {input_path}")
    try:
        model = tf.keras.models.load_model(input_path)
    except Exception as e:
        print(f"Error cargando el .keras: {e}")
        return

    print("2. Exportando a SavedModel...")
    model.export(temp_dir)
    
    print("3. Configurando convertidor hibrido (necesario para LSTMs)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
    
    # Permitir operaciones nativas de TensorFlow (Select TF Ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Operaciones estándar (ligeras)
        tf.lite.OpsSet.SELECT_TF_OPS    # Operaciones complejas de TF (necesarias para LSTM dinámico)
    ]
    
    #Desactivar la conversión forzada de listas 
    converter._experimental_lower_tensor_list_ops = False
    
    #Habilitar variables de recursos
    converter.experimental_enable_resource_variables = True

    #Optimización estándar
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("4. Convirtiendo a TFLite...")
    try:
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Modelo guardado: {output_path}")
        
    except Exception as e:
        print(f"\nError durante la conversión: {e}")
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


#INPUT_MODEL = "lsm_best_model_dori.keras"
#INPUT_MODEL = "lsm_ultimate_model.keras"
INPUT_MODEL = "ruben/lsm_ultimate_model.keras"
OUTPUT_MODEL = "lsm_best_model_android.tflite"
    
convert_lstm_model(INPUT_MODEL, OUTPUT_MODEL)