package com.example.clubpath.Motion

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MotionModel(private val context: Context, private val modelPath: String) {
    private var interpreter: Interpreter? = null

    /*
    private val keypoint3dBuffer: TensorBuffer by lazy {
        val probabilityTensorIndex = 0
        val arrayShape =
            interpreter?.getOutputTensor(probabilityTensorIndex)?.shape() // {1, 16, 3}
        val probabilityDataType = interpreter?.getOutputTensor(probabilityTensorIndex)?.dataType()
        TensorBuffer.createFixedSize(arrayShape, probabilityDataType)
    }
    */
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
        val model = loadModelFile(assetManager, modelPath)

        val interpreter = Interpreter(model)
        // TODO: Read the model input shape from model file.

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

    /**
     * Initialize the input objects and fill them with zeros.
     */
    private fun initializeDataInput(): HashMap<String, Any> {
        val inputs = HashMap<String, Any>()
        for (inputName in interpreter!!.getSignatureInputs(SIGNATURE_KEY)) {
            // Initialize a ByteBuffer filled with zeros as an initial input of the TFLite model.
            val tensor = interpreter!!.getInputTensorFromSignature(inputName, SIGNATURE_KEY)

            val byteBuffer = ByteBuffer.allocateDirect(tensor.numBytes())
            byteBuffer.order(ByteOrder.nativeOrder())
            inputs[inputName] = byteBuffer
        }
        return inputs
    }

    /**
     * Initialize the output objects to store the TFLite model outputs.
     */
    private fun initializeOutput(): HashMap<String, Any> {
        val outputs = HashMap<String, Any>()
        for (outputName in interpreter!!.getSignatureOutputs(SIGNATURE_KEY)) {
            // Initialize a ByteBuffer to store the output of the TFLite model.
            val tensor = interpreter!!.getOutputTensorFromSignature(outputName, SIGNATURE_KEY)
            val byteBuffer = ByteBuffer.allocateDirect(tensor.numBytes())
            byteBuffer.order(ByteOrder.nativeOrder())
            outputs[outputName] = byteBuffer
        }

        return outputs
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
        private const val INPUT_NAME = "input"
        private const val OUTPUT_NAME = "output"
        private const val SIGNATURE_KEY = "serving_default"
    }
}