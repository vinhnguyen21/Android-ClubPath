package com.example.clubpath.components3D

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.TaskCompletionSource
import com.google.android.gms.tasks.Task
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class Lifting3DModel(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val keypoint3dBuffer: TensorBuffer by lazy {
        val probabilityTensorIndex = 0
        val arrayShape =
            interpreter?.getOutputTensor(probabilityTensorIndex)?.shape() // {1, 16, 3}
        val probabilityDataType = interpreter?.getOutputTensor(probabilityTensorIndex)?.dataType()
        TensorBuffer.createFixedSize(arrayShape, probabilityDataType)
    }

    var isInitialized = false

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    fun initialize(): Task<Void?> {
        val task = TaskCompletionSource<Void?>()
        executorService.execute {
            try {
                initializeInterpreter()
                task.setResult(null)
            } catch (e: IOException) {
                task.setException(e)
            }
        }
        return task.task
    }

    @Throws(IOException::class)
    private fun initializeInterpreter() {
        // Load the TF Lite model from asset folder and initialize TF Lite Interpreter with NNAPI enabled.
        val assetManager = context.assets
        val model = loadModelFile(assetManager, MODEL_PATH)
        val interpreter = Interpreter(model)

        // TODO: Read the model input shape from model file.

        // Read input shape from model file.
        val inputShape = interpreter.getInputTensor(0).shape()

        // Finish interpreter initialization.
        this.interpreter = interpreter

        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {
        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        executorService.execute {
            interpreter?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    private fun init3DInput(keypoint: INDArray): ByteBuffer {
        // Convert the ND4J array to a DataBuffer
        val dataBuffer: DataBuffer = keypoint.data()

        // Get the underlying float buffer from the DataBuffer
        val floatBuffer: FloatArray = dataBuffer.asFloat()
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 27, 16, 2), DataType.FLOAT32)
        inputFeature0.loadArray(floatBuffer)

        return inputFeature0.buffer
    }

    fun classify(inPutArray: INDArray): FloatArray {
        val inputModel = init3DInput(inPutArray)
        Log.d("3D", "stand here")
        interpreter?.run(inputModel, keypoint3dBuffer.buffer.rewind())
        return keypoint3dBuffer.floatArray
    }

//    fun classifyAsync(inPutArray: INDArray): Task<FloatArray> {
//        val task = TaskCompletionSource<String>()
//        executorService.execute {
//            val result = classify(inPutArray)
//            task.setResult(result)
//        }
//        return task.task
//    }

    companion object {
        private val TAG = Lifting3DModel::class.java.simpleName
        // ClassifierFloatEfficientNet model
        private const val MODEL_PATH = "modelLifting.tflite"
    }
}