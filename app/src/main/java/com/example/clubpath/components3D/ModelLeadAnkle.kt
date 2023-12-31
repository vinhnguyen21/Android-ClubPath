package com.example.clubpath.components3D

import android.content.Context
import com.example.clubpath.Motion.MotionHelperUtils
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

class ModelLeadAnkle (private val context: Context, private val modelPath: String): Closeable {
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

    private fun processLeadAnkleInput(keypoint: INDArray): ByteBuffer {
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

    fun classify(inPutArray: INDArray): INDArray {
        val totalFrame = inPutArray.shape()[0].toDouble()

        val outputShape = IntArray(2)
        outputShape[0] = 360
        outputShape[1] = 1

        //==== Todo
        /// Do linspace here to forward the result from n frames -> 360 frames -> n frames
        val idxMappingInput = MotionHelperUtils().linspace(0.0, totalFrame - 1.0, 360)
        val mappingIndice = NDArrayIndex.indices(*idxMappingInput.toLongArray())
        val inputMotion = inPutArray.get(mappingIndice, NDArrayIndex.all())

        var outputs = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        val inputModel = processLeadAnkleInput(inputMotion)

        // Run inference using the TFLite model.
        interpreter.run(inputModel, outputs.buffer.rewind())

        // mapping back Index
        var tmpOutput = Nd4j.create(outputs.floatArray)
        tmpOutput = tmpOutput.reshape(outputShape[0].toLong(), outputShape[1].toLong())

        val idxMappingBackInput = MotionHelperUtils().linspace(0.0, 360.0 - 1.0, totalFrame.toInt())
        val finalOutput = tmpOutput.get(
            NDArrayIndex.indices(*idxMappingBackInput.toLongArray()),
            NDArrayIndex.all()
        )

        return finalOutput
    }

    companion object {
        private val TAG = ModelLeadAnkle::class.java.simpleName
    }
}