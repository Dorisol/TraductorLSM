Este proyecto implementa un traductor automático de LSM que captura movimientos corporales y de manos a través de una cámara, procesándolos con MediaPipe y clasificándolos mediante una red neuronal LSTM entrenada. El sistema está disponible en dos plataformas:

- Versión Desktop (Python): Para pruebas y desarrollo con cámara web
- Versión Móvil (Android): Aplicación nativa para uso portátil

#### Caracteristicas
- Reconocimiento de 22 palabras en LSM
- Detección en tiempo real 
- Normalización de datos independiente de distancia a la cámara
- Sistema de estabilización de predicciones

#### Uso de la aplicación móvil
1. Abrir aplicación
2. Presionar "Traducir" para iniciar reconocimiento
3. Realizar señas frente a la cámara frontal
4. Las palabras detectadas aparecen en pantalla
5. Presionar "Pausar" para detener
6. Presionar "Limpiar" para borrar texto
