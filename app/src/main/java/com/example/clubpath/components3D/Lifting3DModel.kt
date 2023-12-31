package com.example.clubpath.components3D

import android.content.Context
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer

class Lifting3DModel(context: Context) : Closeable {
    private val numThread: Int = 4
    private val interpreterInitializer = lazy {
        val interpreterOption = InterpreterApi.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            .setNumThreads(numThread)
        InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOption)
    }

    private val interpreter: InterpreterApi by interpreterInitializer
    var isInitialized = interpreterInitializer.isInitialized()

    // Output probability TensorBuffer
    private val keypoint3dBuffer: TensorBuffer by lazy {
        val probabilityTensorIndex = 0
        val arrayShape =
            interpreter.getOutputTensor(probabilityTensorIndex).shape() // {1, 16, 3}
        val probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType()
        TensorBuffer.createFixedSize(arrayShape, probabilityDataType)
    }

    /** Releases TFLite resources if initialized. */
    override fun close() {
        if (interpreterInitializer.isInitialized()) {
            interpreter.close()
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

    fun classify(inPutArray: INDArray): INDArray {
        val inputModel = init3DInput(inPutArray)
        interpreter?.run(inputModel, keypoint3dBuffer.buffer.rewind())

        // convert to INDArray
        var output = Nd4j.create(keypoint3dBuffer.floatArray)

        output = output.reshape(1, 16, 3)
        return output
    }

    companion object {
        private val TAG = Lifting3DModel::class.java.simpleName

        // ClassifierFloatEfficientNet model
        private const val MODEL_PATH = "modelLifting.tflite"
    }
}