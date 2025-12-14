package com.example.traductorlsm

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var resultsPose: PoseLandmarkerResult? = null
    private var resultsHand: HandLandmarkerResult? = null

    // Dimensiones de la imagen procesada (para escalar correctamente)
    private var imageHeight = 1
    private var imageWidth = 1

    // Configuración de pinceles
    private val pointPaint = Paint().apply {
        color = Color.YELLOW
        strokeWidth = 12f
        style = Paint.Style.FILL
    }

    private val linePaint = Paint().apply {
        color = Color.CYAN
        strokeWidth = 8f
        style = Paint.Style.STROKE
    }

    // Actualizar resultados y redibujar
    fun setResults(
        pose: PoseLandmarkerResult?,
        hands: HandLandmarkerResult?,
        imgHeight: Int,
        imgWidth: Int
    ) {
        resultsPose = pose
        resultsHand = hands
        this.imageHeight = imgHeight
        this.imageWidth = imgWidth

        // Importante: forzar redibujado en el hilo principal
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Factor de escala para ajustar coordenadas normalizadas a la pantalla
        val scaleX = width.toFloat() // Como son normalizadas (0 a 1), multiplicamos por ancho de vista
        val scaleY = height.toFloat()

        // --- DIBUJAR POSE ---
        resultsPose?.let { poseResult ->
            if (poseResult.landmarks().isNotEmpty()) {
                val landmarks = poseResult.landmarks()[0]

                // Dibujar puntos
                for (landmark in landmarks) {
                    canvas.drawPoint((1 - landmark.x()) * scaleX, landmark.y() * scaleY, pointPaint)
                }

                // Dibujar líneas del cuerpo (Conexiones básicas)
                // Brazos
                drawLine(canvas, landmarks[11], landmarks[13], scaleX, scaleY) // Hombro I -> Codo I
                drawLine(canvas, landmarks[13], landmarks[15], scaleX, scaleY) // Codo I -> Muñeca I
                drawLine(canvas, landmarks[12], landmarks[14], scaleX, scaleY) // Hombro D -> Codo D
                drawLine(canvas, landmarks[14], landmarks[16], scaleX, scaleY) // Codo D -> Muñeca D
                // Torso
                drawLine(canvas, landmarks[11], landmarks[12], scaleX, scaleY) // Hombro I -> Hombro D
            }
        }

        // --- DIBUJAR MANOS ---
        resultsHand?.let { handResult ->
            for (landmarks in handResult.landmarks()) {
                // Dibujar puntos de la mano
                for (landmark in landmarks) {
                    canvas.drawPoint((1 - landmark.x()) * scaleX, landmark.y() * scaleY, pointPaint)
                }

                // Dibujar conexiones de dedos
                // Pulgar
                drawLine(canvas, landmarks[0], landmarks[1], scaleX, scaleY)
                drawLine(canvas, landmarks[1], landmarks[2], scaleX, scaleY)
                drawLine(canvas, landmarks[2], landmarks[3], scaleX, scaleY)
                drawLine(canvas, landmarks[3], landmarks[4], scaleX, scaleY)
                // Índice
                drawLine(canvas, landmarks[0], landmarks[5], scaleX, scaleY)
                drawLine(canvas, landmarks[5], landmarks[6], scaleX, scaleY)
                drawLine(canvas, landmarks[6], landmarks[7], scaleX, scaleY)
                drawLine(canvas, landmarks[7], landmarks[8], scaleX, scaleY)
                // Medio
                drawLine(canvas, landmarks[9], landmarks[10], scaleX, scaleY)
                drawLine(canvas, landmarks[10], landmarks[11], scaleX, scaleY)
                drawLine(canvas, landmarks[11], landmarks[12], scaleX, scaleY)
                // Anular
                drawLine(canvas, landmarks[13], landmarks[14], scaleX, scaleY)
                drawLine(canvas, landmarks[14], landmarks[15], scaleX, scaleY)
                drawLine(canvas, landmarks[15], landmarks[16], scaleX, scaleY)
                // Meñique
                drawLine(canvas, landmarks[17], landmarks[18], scaleX, scaleY)
                drawLine(canvas, landmarks[18], landmarks[19], scaleX, scaleY)
                drawLine(canvas, landmarks[19], landmarks[20], scaleX, scaleY)
                // Palma
                drawLine(canvas, landmarks[5], landmarks[9], scaleX, scaleY)
                drawLine(canvas, landmarks[9], landmarks[13], scaleX, scaleY)
                drawLine(canvas, landmarks[13], landmarks[17], scaleX, scaleY)
                drawLine(canvas, landmarks[0], landmarks[17], scaleX, scaleY)
            }
        }
    }

    private fun drawLine(
        canvas: Canvas,
        start: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        end: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        scaleX: Float,
        scaleY: Float
    ) {
        val startX = (1 - start.x()) * scaleX
        val startY = start.y() * scaleY
        val endX = (1 - end.x()) * scaleX
        val endY = end.y() * scaleY

        canvas.drawLine(startX, startY, endX, endY, linePaint)
    }
}