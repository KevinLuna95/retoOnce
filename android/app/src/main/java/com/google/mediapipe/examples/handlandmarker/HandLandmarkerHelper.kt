/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.google.mediapipe.examples.handlandmarker.MainActivity

class TFLiteModel(context: Context) {
    private var interpreter: Interpreter? = null
    private var landmarkBuffer: MutableList<List<NormalizedLandmark>> = mutableListOf()
    init {
        try {
            val model = FileUtil.loadMappedFile(context, "ml/modelo.tflite")
            interpreter = Interpreter(model)
        } catch (e: Exception) {
            Log.e("TFLiteModel", "Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }

    fun runModel(inputData: MutableList<MutableList<NormalizedLandmark>>?): FloatArray {
        // Verificar inputData
        if (inputData.isNullOrEmpty()) {
            Log.e("TFLiteModel", "Input data is null or empty")
            return FloatArray(0)
        }

        val flatLandmarks = inputData.flatten()
        if (flatLandmarks.size != 21 && flatLandmarks.size != 42) {
            Log.e("TFLiteModel", "${flatLandmarks.size} Input data does not match the expected shape: [10, 126]")
            return FloatArray(0)
        }

        landmarkBuffer.add(flatLandmarks)
        if (landmarkBuffer.size > 10) {
            landmarkBuffer.removeAt(0)
        }

        val inputByteBuffer = convertLandmarksToByteBuffer(landmarkBuffer)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(0)
        val outputShape = interpreter?.getOutputTensor(0)?.shape() ?: intArrayOf(0)

        Log.d("TFLiteModel", "Input shape: ${inputShape.contentToString()}")
        Log.d("TFLiteModel", "Output shape: ${outputShape.contentToString()}")

        val inputBuffer = TensorBuffer.createFixedSize(inputShape, interpreter?.getInputTensor(0)?.dataType())
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, interpreter?.getOutputTensor(0)?.dataType())

        inputBuffer.loadBuffer(inputByteBuffer)

        return try {
            interpreter?.run(inputBuffer.buffer, outputBuffer.buffer)
            outputBuffer.floatArray
        } catch (e: Exception) {
            Log.e("TFLiteModel", "Error running model: ${e.message}")
            e.printStackTrace()
            FloatArray(0)
        }
    }
}

fun convertLandmarksToByteBuffer(landmarks: MutableList<List<NormalizedLandmark>>): ByteBuffer {
    // El tamaño esperado es de 10 listas, cada una con 126 elementos (42 landmarks x 3 valores float cada uno)
    val expectedSize = 10 * 126

    for (landmarkSet in landmarks) {
        if (landmarkSet.size != 21 && landmarkSet.size != 42) {
            throw IllegalArgumentException("Input data does not match the expected shape: [10, 126]")
        }
    }

    val byteBuffer = ByteBuffer.allocateDirect(expectedSize * 4).order(ByteOrder.nativeOrder())

    // 10 Frames
    for (n in 0..9) {
        if (n < landmarks.size) {
            // 42 Landmarks Per Frame (126 Points)
            for (i in 0..41)  {
                if (i < landmarks[n].size)  {
                    val landmark = landmarks[n][i]
                    byteBuffer.putFloat(landmark.x())
                    byteBuffer.putFloat(landmark.y())
                    byteBuffer.putFloat(landmark.z())
                }
                else {
                    byteBuffer.putFloat(0F)
                    byteBuffer.putFloat(0F)
                    byteBuffer.putFloat(0F)
                }
            }
        }
        else {
            for (i in 0..125)  {
                byteBuffer.putFloat(0F)
            }
        }
    }

    byteBuffer.rewind()
    return byteBuffer
}

public class HandLandmarkerHelper(
    var minHandDetectionConfidence: Float = DEFAULT_HAND_DETECTION_CONFIDENCE,
    var minHandTrackingConfidence: Float = DEFAULT_HAND_TRACKING_CONFIDENCE,
    var minHandPresenceConfidence: Float = DEFAULT_HAND_PRESENCE_CONFIDENCE,
    var maxNumHands: Int = DEFAULT_NUM_HANDS,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val handLandmarkerHelperListener: LandmarkerListener? = null
) {
    // For this example this needs to be a var so it can be reset on changes.
    // If the Hand Landmarker will not change, a lazy val would be preferable.
    private var handLandmarker: HandLandmarker? = null

    private var signPredictionTreshold: Float = 0.25F
    private var prevSign: String = ""

    init {
        setupHandLandmarker()
    }

    fun clearHandLandmarker() {
        handLandmarker?.close()
        handLandmarker = null
    }

    // Return running status of HandLandmarkerHelper
    fun isClose(): Boolean {
        return handLandmarker == null
    }

    // Initialize the Hand landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupHandLandmarker() {
        // Set general hand landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_HAND_LANDMARKER_TASK)

        // Check if runningMode is consistent with handLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (handLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "handLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Hand Landmarker.
            val optionsBuilder =
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinHandDetectionConfidence(minHandDetectionConfidence)
                    .setMinTrackingConfidence(minHandTrackingConfidence)
                    .setMinHandPresenceConfidence(minHandPresenceConfidence)
                    .setNumHands(maxNumHands)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            handLandmarker =
                HandLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            handLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            handLandmarkerHelperListener?.onError(
                "Hand Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to HandlandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
    }

    fun obtenerPosicionMaximoValor(array: FloatArray): Int {
        var maximoIndice = 0 // Inicializamos el índice del máximo valor como 0
        var maximoValor = array[0] // Inicializamos el máximo valor como el primer elemento del array

        // Recorremos el array desde la segunda posición
        for (i in 1 until array.size) {
            if (array[i] > maximoValor) { // Si encontramos un valor mayor
                maximoValor = array[i] // Actualizamos el máximo valor
                maximoIndice = i // Actualizamos el índice del máximo valor
            }
        }

        return maximoIndice
    }
    fun handleModelOutput(outputData: FloatArray) {
        // Imprimir la salida del modelo para depuración
        val res = obtenerPosicionMaximoValor(outputData)
        val arr = arrayOf("a","b","c","i","n")
        if (outputData[res] >= signPredictionTreshold && res < arr.size) {
            val c = arr[res]
            if (c != prevSign) {
                result_output = "$result_output $c"
                Log.d("TFLiteModel-output", "$result_output (${outputData[res]})")
                // Implementa aquí la lógica para manejar los datos de salida del modelo.
                // Por ejemplo, podrías actualizar la UI, enviar los datos a otra parte de tu aplicación, etc.
                prevSign = c
            }
        }
        else {
            prevSign = ""
        }
    }

    // Run hand hand landmark using MediaPipe Hand Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        handLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // hand landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the hand landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val resultList = mutableListOf<HandLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run hand landmarker using MediaPipe Hand Landmarker API
                    handLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->
                            resultList.add(detectionResult)
                        } ?: run{
                        didErrorOccurred = true
                        handLandmarkerHelperListener?.onError(
                            "ResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    handLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ResultBundle(resultList, inferenceTimePerFrameMs, height, width)
        }
    }

    // Accepted a Bitmap and runs hand landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run hand landmarker using MediaPipe Hand Landmarker API
        handLandmarker?.detect(mpImage)?.also { landmarkResult ->
            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                listOf(landmarkResult),
                inferenceTimeMs,
                image.height,
                image.width
            )
        }

        // If handLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        handLandmarkerHelperListener?.onError(
            "Hand Landmarker failed to detect."
        )
        return null
    }

    // Return the landmark result to this HandLandmarkerHelper's caller
    fun returnLivestreamResult(result: HandLandmarkerResult, input: MPImage) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()

        handLandmarkerHelperListener?.onResults(
            HandLandmarkerHelper.ResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width
            )
        )

        val inputData: MutableList<MutableList<NormalizedLandmark>>? = result.landmarks()

        if (inputData != null && inputData.isNotEmpty()) {
            try {
                val tfliteModel = TFLiteModel(context)
                   val outputData = tfliteModel.runModel(inputData)

                if (outputData.isNotEmpty()) {
                   handleModelOutput(outputData)
                } else {
                   // Log.e(TAG, "Model output is empty")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error handling model output: ${e.message}")
                e.printStackTrace()
            }
        } else {
            //Log.e(TAG, "Input data for TFLite model is null or empty")
        }
/*
        result.landmarks().forEachIndexed { handIndex, handLandmarks ->
            Log.d(TAG, "Hand $handIndex:")
            handLandmarks.forEachIndexed { pointIndex, landmark ->
                Log.d(TAG, "Point $pointIndex: (${landmark.x()}, ${landmark.y()}, ${landmark.z()})")
            }
        }*/
    }


    // Return errors thrown during detection to this HandLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        handLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    companion object {
        const val TAG = "HandLandmarkerHelper"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"
        var result_output: String = ""
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_HAND_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_HAND_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_HANDS = 2
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    data class ResultBundle(
        val results: List<HandLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}

