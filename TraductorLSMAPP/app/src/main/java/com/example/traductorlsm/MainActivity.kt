package com.example.traductorlsm

import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.traductorlsm.databinding.LayoutTraductorBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.Manifest
import android.text.method.ScrollingMovementMethod
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var binding: LayoutTraductorBinding
    private lateinit var cameraExecutor: ExecutorService

    //MediaPipe
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null

    //TensorFlow Lite
    private var interpreter: Interpreter? = null
    private val labels = mutableListOf<String>()

    //Control de estados
    private var isTranslating = false
    private val sequenceBuffer = mutableListOf<FloatArray>()
    private val sequenceLength = 45
    private val keypointsSize = 33*4 + 21*3 + 21*3    //258 puntos clave

    private var frameCounter = 0
    private val predictionInterval = 1

    //FPS overlay
    private var fpsCounter = 0
    private var lastFpsTime = System.currentTimeMillis()


    private val predictions = mutableListOf<Int>()      //indice de clases
    private val sentence = mutableListOf<String>()      //Palabras detectadas
    private val THRESHOLD = 0.4f                       //confianza
    private val STABILITY_WINDOW =10                //ventana estabilidad


    private lateinit var overlayView: OverlayView


    companion object {
        private const val TAG = "LSMTranslator"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        binding = LayoutTraductorBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.tvTraduccion.movementMethod = ScrollingMovementMethod()
        overlayView = binding.overlayView


        //Verificar permisos
        if(allPermissionsGranted()){
            iniciarApp()
        }else{
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        configurarBotones()
    }

    private fun iniciarApp(){
        configurarMediaPipe()
        cargarModelo()
        iniciarCamara()
    }

    private fun configurarBotones(){
        binding.btnLectura.setOnClickListener {
            isTranslating = !isTranslating
            if (isTranslating){
                binding.btnLectura.text = "Pausar"
                binding.btnLectura.setIconResource(R.drawable.ic_abc)
                sequenceBuffer.clear()
                predictions.clear()
                sentence.clear()

                //rellenar el buffer con ceros
                val emptyFrame = FloatArray(keypointsSize){0f}
                repeat(sequenceLength){
                    sequenceBuffer.add(emptyFrame)
                }
                Log.d(TAG, "Traduccion iniciada")
            }else{
                binding.btnLectura.text = "Traducir"
                Log.d(TAG, "Traduccion pausada")
            }
        }

        binding.btnLimpiar.setOnClickListener {
            binding.tvTraduccion.text = ""
            binding.tvTraduccion.hint = "Traducción"
            sequenceBuffer.clear()
            predictions.clear()
            sentence.clear()
            Log.d(TAG, "Buffer limpio")
        }
    }

    private fun configurarMediaPipe() {
        try {
            val baseHandOptions = com.google.mediapipe.tasks.core.BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            //Configurar Hand landmarker
            val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseHandOptions)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(this, handOptions)

            //Configurar Pose landmarker
            val basePoseOptions = com.google.mediapipe.tasks.core.BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_lite.task")
                .build()

            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(basePoseOptions)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(this, poseOptions)
        }catch (e: Exception){
            Log.e(TAG, "Error configurando MediaPipe: ${e.message}")
            Toast.makeText(this, "Error al iniciar MediaPipe", Toast.LENGTH_SHORT).show()
        }
    }

    private fun cargarModelo(){
        try{
            Log.d(TAG, "Intentando cargar el modelo TFLite...")

            //Verificar que el archivo existe
            val modelFiles = assets.list("")
            Log.d(TAG, "Archivos en assets: ${modelFiles?.joinToString(", ")}")

            //Cargar modelo TFLite
            val modelFile = cargarArchivoModelo("lsm_best_model_android.tflite")
            Log.d(TAG, "Archivo modelo cargado, tamaño: ${modelFile.capacity()} bytes")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(false)
            }
            interpreter = Interpreter(modelFile, options)

            Log.d(TAG, "Input count: ${interpreter?.inputTensorCount}")
            Log.d(TAG, "Output count: ${interpreter?.outputTensorCount}")

            if (interpreter != null) {
                val inputShape = interpreter!!.getInputTensor(0).shape()
                val outputShape = interpreter!!.getOutputTensor(0).shape()
                Log.d(TAG, "Input shape: ${inputShape.contentToString()}")
                Log.d(TAG, "Output shape: ${outputShape.contentToString()}")
            }

            //Cargar etiquetas desde actions.txt
            assets.open("actions.txt").bufferedReader().use { reader->
                labels.clear()
                reader.forEachLine { line->
                    if (line.isNotBlank()){
                        labels.add(line.trim())
                    }
                }
            }

            Log.d(TAG, "Etiquetas cargadas")
            Log.d(TAG, "Total clases: ${labels.size}")

        }catch (e: Exception){
            Log.e(TAG, "Error cargando Modelo: ${e.message}")
            Toast.makeText(this, "Error al cargar modelo", Toast.LENGTH_SHORT).show()
        }
    }

    private fun cargarArchivoModelo(fileName: String): MappedByteBuffer{
        val fileDescriptor = assets.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @OptIn(ExperimentalGetImage::class)
    private fun iniciarCamara(){
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewViewCamera.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        procesarFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error iniciando cámara: ${e.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    //Rotar la imagen
    private fun rotateBitmap(bitmap: android.graphics.Bitmap, degrees: Int, flipHorizontal: Boolean = false): android.graphics.Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees.toFloat())

        if (flipHorizontal) {
            matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
        }

        return android.graphics.Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }

    @OptIn(ExperimentalGetImage::class)
    private fun procesarFrame(imageProxy: ImageProxy){

        // --- MEDIDOR FPS ---
        fpsCounter++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFpsTime >= 1000) {
            Log.d("FPS_CHECK", "FPS Actuales: $fpsCounter")
            lastFpsTime = currentTime
            fpsCounter = 0
        }

        //Si no esta traduciendo, ignorar
        if (!isTranslating) {
            imageProxy.close()
            return
        }

        //Convertir a bitmap y rotar
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

            val originalBitmap = imageProxy.toBitmap()
            val processedBitmap = rotateBitmap(
                originalBitmap,
                rotationDegrees,
                flipHorizontal = false
            )

            val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(
                processedBitmap
            ).build()


            //Detectar pose
            val poseResult = poseLandmarker?.detect(mpImage)

            //detectar mano
            var handResult: HandLandmarkerResult? = null
            if (poseResult != null && poseResult.landmarks().isNotEmpty()) {
                handResult = handLandmarker?.detect(mpImage)
            }

            /*
            //DIBUJAR LANDMARKS
            val finalPose = poseResult
            val finalHand = handResult
            val imgW = processedBitmap.width
            val imgH = processedBitmap.height

            runOnUiThread {
                overlayView.setResults(finalPose, finalHand, imgH, imgW)
            }

             */

            //Extraer keypoints
            val keypoints = extraerKeypoints(poseResult, handResult)

            if (keypoints != null) {
                //Agregar al buffer
                sequenceBuffer.add(keypoints)

                //Mantener solo los 45 frames
                if (sequenceBuffer.size > sequenceLength){
                    sequenceBuffer.removeAt(0)
                }

                // Predecir cuando el bbuffer esté lleno
                if (sequenceBuffer.size == sequenceLength && frameCounter % predictionInterval == 0){
                    predecirYActualizarFrase()
                }

                frameCounter++
                if (frameCounter > 1000) frameCounter = 0

            } else {
                Log.w(TAG, "No se detectó persona (Pose null)")
            }
        }
        imageProxy.close()   //liberar memoria
    }

    private fun extraerKeypoints(poseResult: PoseLandmarkerResult?, handResult: HandLandmarkerResult?): FloatArray? {
        //validar que hay persona
        if (poseResult == null || poseResult.landmarks().isEmpty()) {
            return null
        }

        val keypoints = FloatArray(keypointsSize)
        var idx = 0

        val landmarks = poseResult.landmarks()[0]

        //Hombros
        val leftShoulder = floatArrayOf(landmarks[11].x(), landmarks[11].y(), landmarks[11].z())
        val rightShoulder = floatArrayOf(landmarks[12].x(), landmarks[12].y(), landmarks[12].z())

        //Punto central
        val centerPoint = floatArrayOf(
            (leftShoulder[0] + rightShoulder[0]) / 2f,
            (leftShoulder[1] + rightShoulder[1]) / 2f,
            (leftShoulder[2] + rightShoulder[2]) / 2f
        )

        //Ancho de hombros. Distancia euclidiana
        var shoulderWidth = sqrt(
            (leftShoulder[0] - rightShoulder[0]) * (leftShoulder[0] - rightShoulder[0]) +
                    (leftShoulder[1] - rightShoulder[1]) * (leftShoulder[1] - rightShoulder[1]) +
                    (leftShoulder[2] - rightShoulder[2]) * (leftShoulder[2] - rightShoulder[2])
        )

        if (shoulderWidth < 0.001f) shoulderWidth = 1.0f

        //Normalizar
        // 1. POSE (33 * 4)
        for (landmark in landmarks) {
            keypoints[idx++] = (landmark.x() - centerPoint[0]) / shoulderWidth
            keypoints[idx++] = (landmark.y() - centerPoint[1]) / shoulderWidth
            keypoints[idx++] = (landmark.z() - centerPoint[2]) / shoulderWidth
            keypoints[idx++] = landmark.visibility().orElse(0f)
        }

        // 2. MANOS
        var leftHandPoints: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? = null
        var rightHandPoints: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? = null

        if (handResult != null && handResult.handedness().isNotEmpty()) {
            for (i in handResult.handedness().indices) {
                val category = handResult.handedness()[i][0]
                val handName = category.displayName()

                if (handName == "Left") {
                    leftHandPoints = handResult.landmarks()[i]
                } else if (handName == "Right") {
                    rightHandPoints = handResult.landmarks()[i]
                }
            }
        }

        // Rellenar Mano Izquierda (21 * 3)
        if (leftHandPoints != null) {
            for (lm in leftHandPoints) {
                keypoints[idx++] = (lm.x() - centerPoint[0]) / shoulderWidth
                keypoints[idx++] = (lm.y() - centerPoint[1]) / shoulderWidth
                keypoints[idx++] = (lm.z() - centerPoint[2]) / shoulderWidth
            }
        } else {
            repeat(21 * 3) { keypoints[idx++] = 0f }
        }

        // Rellenar Mano Derecha (21 * 3)
        if (rightHandPoints != null) {
            for (lm in rightHandPoints) {
                keypoints[idx++] = (lm.x() - centerPoint[0]) / shoulderWidth
                keypoints[idx++] = (lm.y() - centerPoint[1]) / shoulderWidth
                keypoints[idx++] = (lm.z() - centerPoint[2]) / shoulderWidth
            }
        } else {
            repeat(21 * 3) { keypoints[idx++] = 0f }
        }

        return keypoints
    }

    //Predecir
    private fun predecirYActualizarFrase() {
        if (interpreter == null || sequenceBuffer.size < sequenceLength) {
            return
        }

        try {
            // Preparar input (1, 45, 258) ---> 1 secuencia ala vez, 45 frames, 258 keypoints
            val inputArray = Array(1) { Array(sequenceLength) { FloatArray(keypointsSize) } }
            for (i in 0 until sequenceLength) {
                for (j in 0 until keypointsSize) {
                    inputArray[0][i][j] = sequenceBuffer[i][j]
                }
            }

            //output (1, 22)  --> numero de clases
            val outputArray = Array(1) { FloatArray(labels.size) }
            interpreter?.run(inputArray, outputArray)

            //probabilidades
            val probabilities = outputArray[0]
            val bestClassIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: return  //indice de mayor probabilidad
            val confidence = probabilities[bestClassIdx]    //valor de la probabilidad

            // AGREGAR A HISTORIAL
            predictions.add(bestClassIdx)

            //Verificar la estabilidad
            if (predictions.size >= STABILITY_WINDOW) {
                val recentPredictions = predictions.takeLast(STABILITY_WINDOW)

                // Verificar que TODOS sean iguales al mejor
                val isStable = recentPredictions.all { it == bestClassIdx }

                Log.d(TAG, "Top 3 predicciones:")
                val top3Indices = probabilities.indices.sortedByDescending { probabilities[it] }.take(3)
                for (i in top3Indices) {
                    Log.d(TAG, "  ${labels[i]}: ${String.format("%.2f%%", probabilities[i] * 100)}")
                }

                if (isStable && confidence > THRESHOLD) {
                    val currentWord = labels[bestClassIdx]

                    //EVITAR REPETICIONES
                    val shouldAdd = if (sentence.isNotEmpty()) {
                        currentWord != sentence.last()
                    } else {
                        true
                    }

                    //no mostrar el "nada"
                    if (shouldAdd && currentWord != "nada") {
                        sentence.add(currentWord)

                        //limitar a 5 palabras
                        if (sentence.size > 5) {
                            sentence.removeAt(0)
                        }

                        // Actualizar UI
                        runOnUiThread {
                            binding.tvTraduccion.text = sentence.joinToString("\n")
                        }

                        Log.d(TAG, "Palabra detectada: $currentWord (${String.format("%.2f%%", confidence * 100)})")
                    }
                }
            }

            //limpiar historial viejo
            if (predictions.size > 30) {
                predictions.removeAt(0)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error en predicción: ${e.message}")
            e.printStackTrace()
        }
    }

    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                iniciarApp()
            } else {
                Toast.makeText(this, "Permisos no concedidos", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handLandmarker?.close()
        poseLandmarker?.close()
        interpreter?.close()
    }
}