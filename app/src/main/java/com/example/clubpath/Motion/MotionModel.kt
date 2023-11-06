package com.example.clubpath.Motion

import android.content.Context
import android.util.Log
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer

class MotionModel(private val context: Context, private val modelPath: String): Closeable {
    private val numThread: Int = 4
    private val interpreterInitializer = lazy {
        val interpreterOption = InterpreterApi.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            .setNumThreads(numThread)
        InterpreterApi.create(FileUtil.loadMappedFile(context, modelPath), interpreterOption)
    }
    private val interpreter: InterpreterApi by interpreterInitializer

    /** Releases TFLite resources if initialized. */
    override fun close() {
        if (interpreterInitializer.isInitialized()) {
            interpreter.close()
        }
    }

    private fun processMotionInput(keypoint: INDArray): ByteBuffer {
        val inputShape = IntArray(2)
        inputShape[0] = 360
        inputShape[1] = 48

        // Convert the ND4J array to a DataBuffer
        val dataBuffer: DataBuffer = keypoint.data()

        // Get the underlying float buffer from the DataBuffer
        val floatBuffer: FloatArray = dataBuffer.asFloat()
        val inputFeature0 = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        inputFeature0.loadArray(floatBuffer)

        return inputFeature0.buffer
    }

    private fun linspace(start: Double, end: Double, count: Int): ArrayList<Long> {
        require(count > 1) { "Count must be greater than 1" }
        val result: ArrayList<Long> = ArrayList()
        val step = (end - start) / (count - 1)

        for (i in 0 until count) {
            val value = start + i * step
            result.add(value.toLong())
        }

        return result
    }

     fun classify(inPutArray: INDArray): INDArray {
        val totalFrame = inPutArray.shape()[0].toDouble()

        val outputShape = IntArray(2)
        outputShape[0] = 360
        outputShape[1] = 12

        //==== Todo
        /// Do linspace here to forward the result from n frames -> 360 frames -> n frames
        val idxMappingInput = linspace(0.0, totalFrame - 1.0, 360)
        val mappingIndice = NDArrayIndex.indices(*idxMappingInput.toLongArray())
        val inputMotion = inPutArray.get(mappingIndice, NDArrayIndex.all())

        var outputs = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        val inputModel = processMotionInput(inputMotion)

        Log.d("3D", "stand here")
        // Run inference using the TFLite model.
        interpreter?.run(inputModel, outputs.buffer.rewind())

        // mapping back Index
        var tmpOutput = Nd4j.create(outputs.floatArray)
        tmpOutput = tmpOutput.reshape(outputShape[0].toLong(), outputShape[1].toLong())

        val idxMappingBackInput = linspace(0.0, 360.0 - 1.0, totalFrame.toInt())
        val test = tmpOutput.get(
            NDArrayIndex.indices(*idxMappingBackInput.toLongArray()),
            NDArrayIndex.all()
        )
        return test
    }

    companion object {
        private val TAG = MotionModel::class.java.simpleName
    }
}