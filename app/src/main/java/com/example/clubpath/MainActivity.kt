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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import com.example.clubpath.components3D.Lifting3DModel
import com.example.clubpath.ui.theme.ClubPathTheme
import com.example.clubpath.utils.SwingKeypointModel
import com.example.clubpath.components3D.Utils3DHelper
import com.example.clubpath.utils.readJSONFromAssets
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.tasks.await
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.lang.ArithmeticException

class MainActivity : ComponentActivity() {
    private var classifier: Lifting3DModel? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val keypointData = readJSONFromAssets(this, "debugPose.json")

        // === convert json file to keypoint array
        val kptArray: INDArray = convertLstPoseToArray(keypointData) ?: return

        // === Input Parameter
        val totalFrame: Int = keypointData.count()
        val frameWidth: Double = 960.0
        val frameHeight: Double = 1080.0

        setContent {
            ClubPathTheme {
                // A surface container using the 'background' color from the theme
                val context = LocalContext.current
                val useGPU = false
                val tfliteOption = TfLiteInitializationOptions.builder()
                if (useGPU) {
                    tfliteOption.setEnableGpuDelegateSupport(true)
                }
                LaunchedEffect(key1 = Unit) {
                    TfLite.initialize(context, tfliteOption.build())
                        .addOnSuccessListener {
                            Log.d("SUCCESS LOADING MODEL", "TFLite in Play Services initialized successfully.")
                            classifier = Lifting3DModel(context)
                        }
                        .await()

                    Log.d("Predict3D", "Running 3D Flow.")
                    // ============= Predict 3D and PList =============== //
                    val (raw2dH36M, processed2DH36M) = Utils3DHelper().convertCoCoToHuman36M(
                        kptArray,
                        totalFrame,
                        frameWidth = frameWidth,
                        frameHeight = frameHeight
                    )
                    var isSideView: Boolean = false
                    var isLefty: Boolean = false
                    if (raw2dH36M != null && processed2DH36M != null) {
                        isSideView = Utils3DHelper().isSideViewCheck(raw2dH36M, 10)
                        isLefty = Utils3DHelper().isLeftyCheck(raw2dH36M, isSideView, 10)

                        var (outputPoseGlobal3DMatft, upLift2DMatft) = Utils3DHelper().predict3D(raw2dH36M, processed2DH36M, isSideView, isLefty, frameWidth = frameWidth, frameHeight = frameHeight)
                    }
                }
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Greeting("Android")

                }
            }
        }
    }

    override fun onDestroy() {
        classifier?.close()
        super.onDestroy()
    }
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
                kptArray.putScalar(
                    intArrayOf(frameIdx, iPose, 0),
                    poseEle.location.x.toDouble() ?: 0.0
                )
                kptArray.putScalar(
                    intArrayOf(frameIdx, iPose, 1),
                    poseEle.location.y.toDouble() ?: 0.0
                )
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
