package com.example.clubpath

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.clubpath.ui.theme.ClubPathTheme
import com.example.clubpath.utils.CoCoFormat
import com.example.clubpath.utils.Human36M
import com.example.clubpath.utils.SwingKeypointModel
import com.example.clubpath.utils.readJSONFromAssets
import kotlinx.serialization.descriptors.PrimitiveKind
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.NewAxis
import java.lang.ArithmeticException

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val keypointData = readJSONFromAssets(baseContext, "debugPose.json")

        // === convert json file to keypoint array
        val kptArray: INDArray = convertLstPoseToArray(keypointData) ?: return

        // === Input Parameter
        val totalFrame: Int = keypointData.count()
        val frameWidth: Double = 960.0
        val frameHeight: Double = 1080.0


        // ============= Predict 3D and PList =============== //
        val (raw2dH36M, processed2DH36M) = convertCoCoToHuman36M(kptArray, totalFrame, frameWidth = frameWidth, frameHeight = frameHeight)
        var isSideView: Boolean = false
        var isLefty: Boolean = false
        raw2dH36M?.let {
            isSideView = isSideViewCheck(it, 10)
            isLefty = isLeftyCheck(it, isSideView, 10)
        }
        setContent {
            ClubPathTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Greeting("Android")

                }
            }
        }
    }
}

private fun isSideViewCheck(keypoint: INDArray, considerFramed: Int = 10): Boolean {
    val xHead = keypoint.getDouble(considerFramed, Human36M().head, 0)
    val xRightShoulder = keypoint.getDouble(considerFramed, Human36M().rightShoulder, 0)
    val xLeftShoulder = keypoint.getDouble(considerFramed, Human36M().leftShoulder, 0)

    if (xRightShoulder < xHead &&  xHead < xLeftShoulder) { return false }
    return true
}

private fun isLeftyCheck(keypoint: INDArray, isSideView: Boolean, numFrameCheck: Int): Boolean {
    var isLefty: Boolean = false
    val totalFrame: Int = keypoint.shape()[0].toInt()
    val xLeftWrist = keypoint.get(NDArrayIndex.all(), NDArrayIndex.point(Human36M().leftWrist.toLong()), NDArrayIndex.point(0)).dup()
    val xRightWrist = keypoint.get(NDArrayIndex.all(), NDArrayIndex.point(Human36M().rightWrist.toLong()), NDArrayIndex.point(0)).dup()
    val xCenterWrist = xLeftWrist.add(xRightWrist).div(2.0)
    val xLeftHip = keypoint.get(NDArrayIndex.all(), NDArrayIndex.point(Human36M().leftHip.toLong()), NDArrayIndex.point(0)).dup()
    val xRightHip = keypoint.get(NDArrayIndex.all(), NDArrayIndex.point(Human36M().rightHip.toLong()), NDArrayIndex.point(0)).dup()
    val xCenterHip = xLeftHip.add(xRightHip).div(2.0)

    if (isSideView) {
        val xMeanCenterWrist = xCenterWrist.get(NDArrayIndex.interval(0, minOf(numFrameCheck, totalFrame))).mean(0)
        val xMeanCenterHip = xCenterHip.get(NDArrayIndex.interval(0, minOf(numFrameCheck, totalFrame))).mean(0)
        isLefty = xMeanCenterHip.getDouble(0) >= xMeanCenterWrist.getDouble(0)
    } else {
        val minFrameCenterWrist = Nd4j.getExecutioner().execAndReturn(IMin(xCenterWrist)).finalResult.toInt()
        val maxFrameCenterWrist = Nd4j.getExecutioner().execAndReturn(IMax(xCenterWrist)).finalResult.toInt()
        isLefty = minFrameCenterWrist >= maxFrameCenterWrist

    }
    return isLefty
}

private fun convertCoCoToHuman36M(cocoArray: INDArray, totalFrame: Int, frameWidth: Double, frameHeight: Double): Pair<INDArray?, INDArray?> {
    val keypointShape = IntArray(3)
    keypointShape[0] = totalFrame
    keypointShape[1] = 17
    keypointShape[2] = 2
    var kptHuman36M = Nd4j.zeros(keypointShape, DataType.DOUBLE)

    // pelvis is in the middle of l_hip and r_hip
    val leftHip = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().leftHip.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val rightHip = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().rightHip.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val centerHip = leftHip.add(rightHip)
    kptHuman36M.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()).assign(centerHip)

    // thorax is in the middle of l_shoulder and r_shoulder
    val leftShoulder = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().leftShoulder.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val rightShoulder = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().rightShoulder.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val thorax = leftShoulder.add(rightShoulder)
    kptHuman36M.get(NDArrayIndex.all(), NDArrayIndex.point(8), NDArrayIndex.all()).assign(thorax)

    // head is in the middle of l_eye and r_eye
    val leftEye = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().leftEye.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val rightEye = cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(CoCoFormat().rightEye.toLong()), NDArrayIndex.all()).dup().div(2.0)
    val head = leftEye.add(rightEye)
    kptHuman36M.get(NDArrayIndex.all(), NDArrayIndex.point(10), NDArrayIndex.all()).assign(head)

    // spine is in the middle of thorax and pelvis
    val thoraxSecond = thorax.dup().div(2.0)
    val centerHipSecond = centerHip.dup().div(2.0)
    val spine = thoraxSecond.add(centerHipSecond)
    kptHuman36M.get(NDArrayIndex.all(), NDArrayIndex.point(7), NDArrayIndex.all()).assign(spine)

    // mapping other joints
    val h36MMappingIdx = arrayListOf<Int>(1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16)
    val cocoMappingIdx = arrayListOf<Int>(12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10)
    h36MMappingIdx.forEachIndexed {idx, h36mIdx ->
        kptHuman36M.get(NDArrayIndex.all(), NDArrayIndex.point(h36mIdx.toLong()), NDArrayIndex.all())
            .assign(cocoArray.get(NDArrayIndex.all(), NDArrayIndex.point(cocoMappingIdx[idx].toLong()), NDArrayIndex.all()))
    }
    val normedKpt2D = normalizeScreenCoordinate(kptHuman36M, totalFrame, frameWidth, frameHeight)
    return Pair(kptHuman36M, normedKpt2D)
}

private fun normalizeScreenCoordinate(h36mArray: INDArray,
                                      totalFrame: Int,
                                      frameWidth: Double, frameHeight: Double,
                                      numJoint: Int = 17): INDArray {
    val keypointShape = IntArray(3)
    keypointShape[0] = totalFrame
    keypointShape[1] = numJoint
    keypointShape[2] = 2
    var normedKpt2D = Nd4j.zeros(keypointShape, DataType.DOUBLE)
    val heightWidthRatio: Double =  frameHeight / frameWidth
    normedKpt2D.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0))
        .assign(h36mArray.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)).dup().div(frameWidth).mul(2.0).sub(1.0))
    normedKpt2D.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1))
        .assign(h36mArray.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1)).dup().div(frameWidth).mul(2.0).sub(heightWidthRatio))

    return normedKpt2D
}

private fun convertLstPoseToArray(array: List<SwingKeypointModel>): INDArray? {
    try {
        val totalFrame: Int = array.count()
        val keypointShape = IntArray(3)
        keypointShape[0] = totalFrame
        keypointShape[1] = 17
        keypointShape[2] = 2
        var kptArray = Nd4j.zeros(keypointShape, DataType.DOUBLE)
        array.forEachIndexed { frameIdx, frameEle ->
            val ftPose = frameEle.listPose.first()
            ftPose.landmarks.forEachIndexed { iPose, poseEle ->
                kptArray.putScalar(intArrayOf(frameIdx, iPose, 0), poseEle.location.x.toDouble() ?: 0.0)
                kptArray.putScalar(intArrayOf(frameIdx, iPose, 1), poseEle.location.y.toDouble() ?: 0.0)
            }
        }
        return kptArray
    } catch (e: ArithmeticException) {
        Log.e("convert LSTPose Json", "cannot parse JsonKeypoint to array")
    }
    return null
}


@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    ClubPathTheme {
        Greeting("Android")
    }
}