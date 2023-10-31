package com.example.clubpath

import android.content.Context
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
import com.example.clubpath.Motion.MotionModel
import com.example.clubpath.components3D.Lifting3DModel
import com.example.clubpath.components3D.ModelKptsMLP
import com.example.clubpath.components3D.ModelLeadAnkle
import com.example.clubpath.ui.theme.ClubPathTheme
import com.example.clubpath.utils.SwingKeypointModel
import com.example.clubpath.components3D.Utils3DHelper
import com.example.clubpath.utils.readJSONFromAssets
//import com.google.android.gms.tflite.client.TfLiteInitializationOptions
//import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.tasks.await
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.lang.ArithmeticException

class MainActivity : ComponentActivity() {
    private var liftingModel: Lifting3DModel? = null
    // model adjust keypoint MLP
    private var modelKptsMlpFrontRight: ModelKptsMLP? = null
    private var modelKptsMlpFrontLeft: ModelKptsMLP? = null
    private var modelKptsMlpSideRight: ModelKptsMLP? = null
    private var modelKptsMlpSideLeft: ModelKptsMLP? = null
    private var modelKptsMLPChoose: ModelKptsMLP? = null
    private var modelLeadAnkle: ModelLeadAnkle? = null

    private var motionModeltest: MotionModel? = null

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
//                val tfliteOption = TfLiteInitializationOptions.builder()
//                if (useGPU) {
//                    tfliteOption.setEnableGpuDelegateSupport(true)
//                }
                LaunchedEffect(key1 = Unit) {
                    liftingModel = Lifting3DModel(context)
                    liftingModel!!
                        .initialize()
                        .addOnFailureListener { e -> Log.e("LOADING 3D", "Error to setting up 3D Lifting.", e) }
                        .await()
                    //============= Init mlp keypoint models
                    modelKptsMlpFrontRight = initMLPModel(context, "ModelKptsMlpFrontRight.tflite")
                    modelKptsMlpFrontLeft = initMLPModel(context, "ModelKptsMlpFrontLeft.tflite")
                    modelKptsMlpSideRight = initMLPModel(context, "ModelKptsMlpSideRight.tflite")
                    modelKptsMlpSideLeft = initMLPModel(context, "ModelKptsMlpSideLeft.tflite")
                    modelLeadAnkle = initLeadAnkleModel(context, "ModelLeadAnkle.tflite")

                    /*
                    motionModeltest = MotionModel(context, "LstmTinySideMotion_float32.tflite")
                    motionModeltest!!
                        .initialize()
                        .addOnFailureListener { e -> Log.e("LOADING 3D", "Error to setting up 3D Lifting.", e) }
                        .await()
                     */
                    Log.d("Predict3D", "Running 3D Flow.")
                    // ============= Predict 3D and PList =============== //
                    val totalFrame: Int = kptArray.shape()[0].toInt()
                    val utils3DHelper = Utils3DHelper(totalFrame)
                    val (raw2dH36M, processed2DH36M) = utils3DHelper.convertCoCoToHuman36M(
                        kptArray,
                        frameWidth = frameWidth,
                        frameHeight = frameHeight
                    )
                    var isSideView: Boolean = false
                    var isLefty: Boolean = false

                    //=== Todo
                    //= Using tensorflow library not google service
                    if (raw2dH36M != null && processed2DH36M != null && liftingModel!!.isInitialized) {
                        isSideView = utils3DHelper.isSideViewCheck(raw2dH36M, 10)
                        isLefty = utils3DHelper.isLeftyCheck(raw2dH36M, isSideView, 10)

                        /*
                        //======== test motion
                        val keypointShapeTest = IntArray(2)
                        keypointShapeTest[0] = 378
                        keypointShapeTest[1] = 48
                        var dataTest = Nd4j.zeros(keypointShapeTest, DataType.DOUBLE).add(0.01)
                        val resultTest = motionModeltest?.classify(dataTest)
                        print(resultTest)
                        */

                        // choose MLP model following side && handness
                        if (!isSideView && !isLefty) {
                            modelKptsMLPChoose = modelKptsMlpFrontRight
                        } else if (!isSideView && isLefty) {
                            modelKptsMLPChoose = modelKptsMlpFrontLeft
                        } else if (isSideView && !isLefty) {
                            modelKptsMLPChoose = modelKptsMlpSideRight
                        } else {
                            modelKptsMLPChoose = modelKptsMlpSideLeft
                        }
                        var (outputPoseGlobal3DMatft, upLift2DMatft) = utils3DHelper.predict3D(raw2dH36M, processed2DH36M,
                            liftingModel,
                            modelKptsMLPChoose,
                            modelLeadAnkle,
                            isSideView, isLefty,
                            frameWidth = frameWidth, frameHeight = frameHeight)
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

    suspend fun initMLPModel(context: Context, modelPath: String): ModelKptsMLP {
        val model = ModelKptsMLP(context, modelPath)
        model
            .initialize()
            .addOnFailureListener { e -> Log.e("LOADING MLP", "Error to setting up 3D Lifting.", e) }
            .await()
        return model
    }

    suspend fun initLeadAnkleModel(context: Context, modelPath: String): ModelLeadAnkle {
        val model = ModelLeadAnkle(context, modelPath)
        model
            .initialize()
            .addOnFailureListener { e -> Log.e("LOADING Ankle", "Error to setting up 3D Lifting.", e) }
            .await()
        return model
    }

    override fun onDestroy() {
        liftingModel?.close()
        modelKptsMlpFrontLeft?.close()
        modelKptsMlpFrontRight?.close()
        modelKptsMlpSideLeft?.close()
        modelKptsMlpSideRight?.close()
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
