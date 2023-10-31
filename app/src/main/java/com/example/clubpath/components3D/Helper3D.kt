package com.example.clubpath.components3D

import android.util.Log
import com.example.clubpath.Motion.MotionHelperUtils
import com.example.clubpath.utils.CoCoFormat
import com.example.clubpath.utils.Human36M
import com.example.clubpath.utils.UpLift
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms
import kotlin.system.measureTimeMillis

class Utils3DHelper(private val totalFrame: Int) {

    fun isSideViewCheck(keypoint: INDArray, considerFramed: Int = 10): Boolean {
        val xHead = keypoint.getDouble(considerFramed, Human36M().head, 0)
        val xRightShoulder = keypoint.getDouble(considerFramed, Human36M().rightShoulder, 0)
        val xLeftShoulder = keypoint.getDouble(considerFramed, Human36M().leftShoulder, 0)

        if (xRightShoulder < xHead &&  xHead < xLeftShoulder) { return false }
        return true
    }

    fun isLeftyCheck(keypoint: INDArray, isSideView: Boolean, numFrameCheck: Int): Boolean {
        var isLefty: Boolean = false
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

    fun convertCoCoToHuman36M(cocoArray: INDArray, frameWidth: Double, frameHeight: Double): Pair<INDArray?, INDArray?> {
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
        val normedKpt2D = normalizeScreenCoordinate(kptHuman36M, frameWidth, frameHeight)
        return Pair(kptHuman36M, normedKpt2D)
    }

    fun normalizeScreenCoordinate(h36mArray: INDArray,
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

    fun predict3D(rawKptHuman36m: INDArray, processedHuman36M: INDArray,
                  liftingModel: Lifting3DModel?,
                  modelMLP: ModelKptsMLP?,
                  modelLeadAnkle: ModelLeadAnkle?,
                  isSideView: Boolean, isLefty: Boolean,
                  frameWidth: Double, frameHeight: Double): Pair<INDArray?, INDArray?> {
        if (liftingModel == null) { return Pair(null, null) }
        val keypointShape = IntArray(3)
        keypointShape[0] = totalFrame
        keypointShape[1] = 16
        keypointShape[2] = 2
        val upLift2D = mappingUpliftOrder(rawKptHuman36m, keypointShape)
        var videoUpLiftKpts3D = Nd4j.zeros(totalFrame, 16, 3)

        // ================== TOO SLOW ================== //
        var updatedPose3D: INDArray?
        var timeAdjustKpt = measureTimeMillis {
            for (frameIndex in 0 until totalFrame) {
                val poseSequence2D = extractPoseSequence(processedHuman36M, frameIndex)
                val result3D = liftingModel.classify(poseSequence2D!!)
                //===== TODO
                // adjust 3d keypoints
                var inputMLPRegressor: INDArray?
                val timeMLP = measureTimeMillis {
                    inputMLPRegressor =
                        preProcessModelRegressor(result3D!!, isSideView, isLefty)
                }
                val resultMLP = modelMLP?.classify(inputMLPRegressor!!)
                videoUpLiftKpts3D[NDArrayIndex.point(frameIndex.toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                    .assign(resultMLP)

                Log.d("ONLY MLP", "Time $timeMLP ms")
            }


            // ================== TOO SLOW ================== //
            updatedPose3D = postProcessModelRegressor(
                videoUpLiftKpts3D,
                upLift2D,
                modelLeadAnkle,
                isSideView,
                isLefty,
                frameWidth,
                frameHeight
            )
        }
        timeAdjustKpt /= 1000
        Log.d("TOTAL MLP TIME", "----- It took $timeAdjustKpt s")
        return Pair(updatedPose3D, upLift2D)
    }

    private fun postProcessModelRegressor(regressor3D: INDArray, current2DUplift: INDArray,
                                          modelLeadAnkle: ModelLeadAnkle?,
                                            isSideView: Boolean, isLefty: Boolean,
                                            width: Double, height: Double): INDArray? {

        // Normalize smooth 3d
        val globalHumanHeight = MotionHelperUtils().findDeltaAxis(regressor3D, 75)
        val normed3D = regressor3D.dup().div(globalHumanHeight)

        //---- find smooth ankle values
        val normed3DCopy = normed3D.dup().reshape(totalFrame.toLong(), 48)
        val leadAnkleSmoothResult = modelLeadAnkle?.classify(normed3DCopy) // [totalFrame, 1]

        if (leadAnkleSmoothResult != null) {
            return gen3DGlobal(
                normed3D, current2DUplift,
                isSideView, isLefty,
                width, height,
                leadAnkleSmoothResult!!
            )
        }
        return null
    }

    private fun gen3DGlobal(upLift3D: INDArray, uplift2DRaw: INDArray,
                            isSideView: Boolean, isLefty: Boolean,
                            width: Double, height: Double,
                            leadAnklePred: INDArray): INDArray {
        val pixelScaleBackUplift: Double = height / 2.0
        val intArrayShape = IntArray(3)
        intArrayShape[0] = totalFrame
        intArrayShape[1] = 16
        intArrayShape[2] = 2
        var kptHorizonVert3D = Nd4j.zeros(intArrayShape, DataType.DOUBLE)

        if (isSideView) {
            kptHorizonVert3D
                .assign(
                    upLift3D[NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.indices(0, 1)].dup() // take Oz, Oy
                )
        } else {
            kptHorizonVert3D
                .assign(
                    upLift3D[NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.indices(2, 1)].dup()
                )
        }

        //extract xy of ankle and head coordinate from kpts_2d
        val rAnkleXY3D = kptHorizonVert3D[NDArrayIndex.all(), NDArrayIndex.point(UpLift().rightAnkle.toLong()), NDArrayIndex.all()]
        val lAnkleXY3D = kptHorizonVert3D[NDArrayIndex.all(), NDArrayIndex.point(UpLift().leftAnkle.toLong()), NDArrayIndex.all()]
        val midAnkleXY3D = rAnkleXY3D.add(lAnkleXY3D).div(2.0)
        val cShoulderXY3D = kptHorizonVert3D[NDArrayIndex.all(), NDArrayIndex.point(UpLift().centerShoulder.toLong()), NDArrayIndex.all()]

        // extract xy of ankle and head coordinate from kpts_2d
        val rAnkle2D = uplift2DRaw[NDArrayIndex.all(), NDArrayIndex.point(UpLift().rightAnkle.toLong()), NDArrayIndex.all()]
        val lAnkle2D = uplift2DRaw[NDArrayIndex.all(), NDArrayIndex.point(UpLift().leftAnkle.toLong()), NDArrayIndex.all()]
        val midAnkle2D = rAnkle2D.add(lAnkle2D).div(2.0)
        val cShoulder2D = uplift2DRaw[NDArrayIndex.all(), NDArrayIndex.point(UpLift().centerShoulder.toLong()), NDArrayIndex.all()]

        val lSa = cShoulderXY3D.dup().sub(midAnkleXY3D).norm1(1)//Transforms.euclideanDistance(cShoulderXY3D, midAnkleXY3D)
        val lSaPixel = cShoulder2D.dup().sub(midAnkle2D).norm1(1)//Transforms.euclideanDistance(cShoulder2D, midAnkle2D)

        val scale = lSaPixel.div(lSa.add(0.000001))
//        val tmpScale = scale.reshape(totalFrame.toLong(), 1, 1)
        var kpts3DZoom = Nd4j.zeros(totalFrame, 16, 3)
        for (frameIdx in 0 until totalFrame) {
            kpts3DZoom[NDArrayIndex.point(frameIdx.toLong())]
                .assign(upLift3D[NDArrayIndex.point(frameIdx.toLong())]
                            .mul(scale[NDArrayIndex.point(frameIdx.toLong())]))
        }
         //.mulColumnVector()
        val kpts3DXYZoom = kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.indices(2, 1)]
        val kpts3DZYZoom = kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.indices(0, 1)]

        val leadAnkle2D = if (isLefty) {
            rAnkle2D
        } else {
            lAnkle2D
        }

        val leadAnkleXY3DZoom = if (isLefty) {
            kpts3DXYZoom[NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()]
        } else {
            kpts3DXYZoom[NDArrayIndex.all(), NDArrayIndex.point(5), NDArrayIndex.all()]
        }

        val leadAnkleZ3DZoom = if (isLefty) {
            kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()]
        } else {
            kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(5), NDArrayIndex.all()]
        } // get axis 0

        val leadAnkleZY3DZoom = if (isLefty) {
            kpts3DZYZoom[NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()]
        } else {
            kpts3DZYZoom[NDArrayIndex.all(), NDArrayIndex.point(5), NDArrayIndex.all()]
        }

        val leadAnkleX3DZoom = if (isLefty) {
            kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all()]
        } else {
            kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(5), NDArrayIndex.all()]
        } //get axis 2 -> Matft need 2 dimension array with stride

        // Output global 3D
        val leadAnkle2DZSignal = leadAnklePred.mul(scale)
        val leadAnkle2DXSignal = leadAnkle2DZSignal.dup()
        // Output global 3D
        val intArrayShape3D = IntArray(3)
        intArrayShape3D[0] = totalFrame
        intArrayShape3D[1] = 16
        intArrayShape3D[2] = 3
        var kpts3DGlobal = Nd4j.zeros(intArrayShape3D, DataType.DOUBLE)

        for (jointIdx in 0 until 16) {
            var tmpX: INDArray?
            var tmpLead2D: INDArray?
            if (isSideView) {
                tmpX = kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(2)]
                            .sub(leadAnkleX3DZoom[NDArrayIndex.all(), NDArrayIndex.point(2)])
                            .add(leadAnkle2DXSignal[NDArrayIndex.all(), NDArrayIndex.point(0)])
                tmpLead2D = kpts3DZYZoom[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.all()]
                                .sub(leadAnkleZY3DZoom)
                                .add(leadAnkle2D)
                tmpLead2D[NDArrayIndex.all(), NDArrayIndex.point(2)]
                    .assign(tmpX)

                kpts3DGlobal[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.all()]
                    .assign(tmpLead2D)
            } else {
                tmpX = kpts3DZoom[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(0)]
                    .sub(leadAnkleZ3DZoom[NDArrayIndex.all(), NDArrayIndex.point(0)])
                    .add(leadAnkle2DZSignal[NDArrayIndex.all(), NDArrayIndex.point(0)])
                kpts3DGlobal[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(0)]
                    .assign(tmpX)

                tmpLead2D = kpts3DXYZoom[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.all()]
                    .sub(leadAnkleXY3DZoom)
                    .add(leadAnkle2D)
                kpts3DGlobal[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(2)]
                    .assign(tmpLead2D[NDArrayIndex.all(), NDArrayIndex.point(0)])
                kpts3DGlobal[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(1)]
                    .assign(tmpLead2D[NDArrayIndex.all(), NDArrayIndex.point(1)])
            }
        }

        kpts3DGlobal = kpts3DGlobal.div(pixelScaleBackUplift + 0.000001)
        return kpts3DGlobal
    }

    private fun preProcessModelRegressor(resultLifting: INDArray, isSideView: Boolean, isLefty: Boolean): INDArray {
        // convert Adaptpose to Human3.6M keypoints, re-add nose joint
        var h36m3D = Nd4j.zeros(1, 17, 3)
        val timeConvert = measureTimeMillis {
            h36m3D[NDArrayIndex.point(0), NDArrayIndex.interval(0, 9), NDArrayIndex.all()]
                .assign(resultLifting[NDArrayIndex.point(0), NDArrayIndex.interval(0, 9), NDArrayIndex.all()])
            h36m3D[NDArrayIndex.point(0), NDArrayIndex.interval(10, 17), NDArrayIndex.all()]
                .assign(resultLifting[NDArrayIndex.point(0), NDArrayIndex.interval(9, 16), NDArrayIndex.all()])

            val noseJoint = h36m3D[NDArrayIndex.point(0), NDArrayIndex.point(8), NDArrayIndex.all()]
                .add(h36m3D[NDArrayIndex.point(0), NDArrayIndex.point(10), NDArrayIndex.all()]).div(2.0)
            h36m3D[NDArrayIndex.point(0), NDArrayIndex.point(9), NDArrayIndex.all()]
                .assign(noseJoint)
        }

        if (!isSideView) {
            val tmpX = h36m3D[NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0)]
            /*
            if (isLefty) {
                tmpX = tmpX.mul(-1.0)
            }
             */
            val tmpZ = h36m3D[NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(2)].mul(-1.0)

            h36m3D[NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0)].assign(tmpZ)
//                h36m3D[NDArrayIndex.point(0), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(1)].assign(tmpY)
            h36m3D[NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(2)].assign(tmpX)
        }

        //3D Human3.6 to Uplift
        val keypointShape = IntArray(3)
        keypointShape[0] = 1
        keypointShape[1] = 16
        keypointShape[2] = 3
        var upLift3D = mappingUpliftOrder(h36m3D, keypointShape)
        upLift3D = upLift3D.reshape(1, 48)

        return upLift3D
    }


    private fun mappingUpliftOrder(h36mArray: INDArray, keypointShape: IntArray): INDArray {
        val upliftOrder = intArrayOf(3,2,1,4,5,6,0,8,10,9,16,15,14,11,12,13)
        var upliftKpt = Nd4j.zeros(keypointShape, DataType.DOUBLE)

        upliftOrder.forEachIndexed { normalIdx, upliftMapIdx ->

            upliftKpt[NDArrayIndex.all(), NDArrayIndex.point(normalIdx.toLong()), NDArrayIndex.all()]
                .assign(
                    h36mArray[NDArrayIndex.all(), NDArrayIndex.point(upliftMapIdx.toLong()), NDArrayIndex.all()]
                )
        }
        return upliftKpt
    }

    private fun extractPoseSequence(
        keypointProcesses: INDArray,
        currentFrameIdx: Int,
        tempFrameLength: Int = 27
    ): INDArray {
        var input2D = Nd4j.ones(tempFrameLength, 17, 2)
        val frameLeft: Int = (tempFrameLength - 1) / 2
        val frameRight = frameLeft

        val padLeft = maxOf(0, frameLeft - currentFrameIdx)
        val padRight = maxOf(0, frameRight - (totalFrame - 1 - currentFrameIdx))

        val startIdx = maxOf(0, currentFrameIdx - frameLeft)
        val endIdx = minOf(totalFrame, currentFrameIdx + frameRight + 1)

        // retrieve sequence frames
        if (padLeft != 0) {
            input2D[NDArrayIndex.interval(0, padLeft.toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                .assign(input2D[NDArrayIndex.interval(0, padLeft.toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                    .mul(keypointProcesses[NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()]))

//            for (leftIdx in 0 until padLeft) {
//                input2D[NDArrayIndex.point(leftIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
//                    .assign(keypointProcesses.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()))
//            }
            input2D[NDArrayIndex.interval((padLeft + startIdx).toLong(), (padLeft + endIdx).toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                .assign(
                    input2D[NDArrayIndex.interval((padLeft + startIdx).toLong(), (padLeft + endIdx).toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                        .mul(keypointProcesses[NDArrayIndex.interval(startIdx.toLong(), endIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all()])
                )

//            for (tmpIdx in startIdx until endIdx) {
//                input2D[NDArrayIndex.point((padLeft + tmpIdx).toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
//                    .assign(keypointProcesses.get(NDArrayIndex.point(tmpIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all()))
//            }
        } else {
            if (padRight != 0) {
                val lastValue =
//                for (rightIdx in 0 until padRight) {
//                    input2D[NDArrayIndex.point((tempFrameLength - rightIdx - 1).toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
//                        .assign(keypointProcesses.get(NDArrayIndex.point((numFrames - 1).toLong()), NDArrayIndex.all(), NDArrayIndex.all()))
//                }
                input2D[NDArrayIndex.interval((tempFrameLength - padRight), tempFrameLength), NDArrayIndex.all(), NDArrayIndex.all()]
                    .assign(
                        input2D[NDArrayIndex.interval((tempFrameLength - padRight), tempFrameLength), NDArrayIndex.all(), NDArrayIndex.all()]
                            .mul(keypointProcesses[NDArrayIndex.point(totalFrame.toLong() - 1), NDArrayIndex.all(), NDArrayIndex.all()])
                    )

            }

            for ((inputIdx, processIdx) in (0 until (tempFrameLength - padRight)).zip(startIdx until endIdx)) {
                input2D[NDArrayIndex.point(inputIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all()]
                    .assign(keypointProcesses[NDArrayIndex.point(processIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all()])
            }
        }

        // remove Nose Joint
        val noseJoint = 9
        var updateArray = Nd4j.zeros(27, 16, 2)
        updateArray[NDArrayIndex.all(), NDArrayIndex.interval(0, noseJoint), NDArrayIndex.all()]
            .assign(input2D[NDArrayIndex.all(), NDArrayIndex.interval(0, noseJoint), NDArrayIndex.all()])
        updateArray[NDArrayIndex.all(), NDArrayIndex.interval(noseJoint, 16), NDArrayIndex.all()]
            .assign(input2D[NDArrayIndex.all(), NDArrayIndex.interval(noseJoint + 1, 17), NDArrayIndex.all()])
        return updateArray
    }
}
