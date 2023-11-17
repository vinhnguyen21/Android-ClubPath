package com.example.clubpath.autotrimswing

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer

class SwingClassifier(context: Context): Closeable {
    private val numThread: Int = 4
    private val bonePairs = listOf( Pair(0, 0), Pair(1, 0), Pair(2, 0), Pair(3, 1), Pair(4, 2), Pair(5, 0),
        Pair(6, 0), Pair(7, 5), Pair(8, 6), Pair(9, 7), Pair(10, 8), Pair(11, 0),
        Pair(12, 0), Pair(13, 11), Pair(14, 12), Pair(15, 13), Pair(16, 14))

    private val acceptedNumFrames: Int = 90

    private val interpreterInitializer = lazy {
        val interpreterOption = InterpreterApi.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            .setNumThreads(numThread)
        InterpreterApi.create(FileUtil.loadMappedFile(context, MODEL_PATH), interpreterOption)
    }

    private val interpreter: InterpreterApi by interpreterInitializer
    var isInitialized = interpreterInitializer.isInitialized()

    // Output probability TensorBuffer
    private val swingClassBuffer: TensorBuffer by lazy {
        val probabilityTensorIndex = 0
        val arrayShape =
            interpreter.getOutputTensor(probabilityTensorIndex).shape() // {1,5}
        val probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType()
        TensorBuffer.createFixedSize(arrayShape, probabilityDataType)
    }

    override fun close() {
        if (interpreterInitializer.isInitialized()) {
            interpreter.close()
        }
    }

    private fun preprocessSwing(keypoint: Array<Array<Float>>,
                                frameWidth: Float, frameHeight: Float): ByteBuffer {

        val processedKpt = FloatArray(acceptedNumFrames * 17 * 3)
        var dataIndex = 0
//        val processedKpt = Array(acceptedNumFrames) { arrayOf(0.0f) }
        keypoint.forEachIndexed {frameIdx, joints ->
            var tmpJoints = joints.clone()
            for (jointIdx in 0 until (17 * 3) step 3) {
                tmpJoints[jointIdx] = (joints[jointIdx] - frameWidth / 2.0f) / (frameWidth / 2.0f)
                tmpJoints[jointIdx + 1] = (joints[jointIdx + 1] - frameHeight / 2.0f) / (frameHeight / 2.0f)
            }

            // processing following bone
            val boneConnection = boneProcess(tmpJoints)
            boneConnection.forEach {joint ->
                processedKpt[dataIndex++] = joint
            }
        }
        // convert to Tensor
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 90, 17, 3), DataType.FLOAT32)
        inputFeature0.loadArray(processedKpt)
        return inputFeature0.buffer
    }

    private fun boneProcess(normKeypoints: Array<Float>): Array<Float> {
        val boneConnection = Array<Float>(17 * 3) {0.0f}
        bonePairs.forEach { tempPair ->
            boneConnection[tempPair.first * 3] = normKeypoints[tempPair.first * 3] - normKeypoints[tempPair.second * 3] // x
            boneConnection[tempPair.first * 3 + 1] = normKeypoints[tempPair.first * 3 + 1] - normKeypoints[tempPair.second * 3 + 1] // y
            boneConnection[tempPair.first * 3 + 2] = (normKeypoints[tempPair.first * 3 + 2] + normKeypoints[tempPair.second * 3 + 2]) / 2.0f // z
        }
        return boneConnection
    }

    fun classify(keypoints: Array<Array<Float>>, frameWidth: Float, frameHeight: Float): FloatArray {
        //make sure 90 frames keypoints
        if (keypoints.size != acceptedNumFrames) {
            return FloatArray(5)
        }
        val inputModel = preprocessSwing(keypoint = keypoints, frameWidth = frameWidth, frameHeight = frameHeight)
        interpreter?.run(inputModel, swingClassBuffer.buffer.rewind())
        return swingClassBuffer.floatArray
    }

    companion object {
        private val TAG = SwingClassifier::class.java.simpleName

        // ClassifierFloatEfficientNet model
        private const val MODEL_PATH = "STGCN.tflite"
    }
}