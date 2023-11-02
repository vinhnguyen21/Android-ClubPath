package com.example.clubpath.Motion

import com.example.clubpath.utils.UpLift
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.system.measureTimeMillis

class MotionPredictor {
    private var totalFrame: Int? = null

    fun predictMotion(inputGlobal3D: INDArray,
                      modelMotion: MotionModel?,
                      isLefty: Boolean, isSideView: Boolean): Triple<INDArray?, Int?, Int?> {
        totalFrame = inputGlobal3D.shape()[0].toInt()
        val globalHumanHeight = MotionHelperUtils().findDeltaAxis(inputGlobal3D, p1 = 75)

        // move global point to local center-hip
        var localizedPose3D = moveStaticPointToCenterHip(inputGlobal3D)

        //== process input for motion
        //=== Normalize
        localizedPose3D.divi(globalHumanHeight)
        var selectedNormedJoints = extractDesiredJoints(localizedPose3D)
        var videoKinematicFeature = Nd4j.zeros(totalFrame!!.toLong(), 12)
        var inputMotion = Nd4j.zeros(totalFrame!!.toLong(), 48)
        localizedPose3D = localizedPose3D.reshape(totalFrame!!.toLong(), 16 * 3)

        var oHipZeroFrame: INDArray? = null
        var oShoulderZeroFrame: INDArray? = null
        var timeProcessData = measureTimeMillis {
            for (frameIdx in 0 until totalFrame!!) {
                //== init zeroOpenHip && zeroOpenShoulder
                if (frameIdx == 0) {
                    val rShoulder = localizedPose3D[NDArrayIndex.point(0), NDArrayIndex.interval(36, 39)]
                    val lShoulder = localizedPose3D[NDArrayIndex.point(0), NDArrayIndex.interval(39, 42)]
                    val rHip = localizedPose3D[NDArrayIndex.point(0), NDArrayIndex.interval(6, 9)]
                    val lHip = localizedPose3D[NDArrayIndex.point(0), NDArrayIndex.interval(9, 12)]
                    oShoulderZeroFrame = rShoulder.add(lShoulder).div(2.0)
                    oHipZeroFrame = rHip.add(lHip).div(2.0)
                }

                val kinematicFeature = processKinematics(localizedPose3D[NDArrayIndex.point(frameIdx.toLong())], isLefty,
                    oHipZeroFrame!!, oShoulderZeroFrame!!)
                videoKinematicFeature[NDArrayIndex.point(frameIdx.toLong())].assign(kinematicFeature)
            }

            inputMotion[NDArrayIndex.all(), NDArrayIndex.interval(0, 36)]
                .assign(selectedNormedJoints)
            inputMotion[NDArrayIndex.all(), NDArrayIndex.interval(36, 48)]
                .assign(videoKinematicFeature)
        }
        //===== extract raw motion values
        val shoulderTurnRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(5)].dup()
        val shoulderTiltRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(3)].dup()
        val shoulderBendRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(4)].dup()
        val hipTurnRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(11)].dup()
        val hipTiltRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(9)].dup()
        val hipBendRaw = videoKinematicFeature[NDArrayIndex.all(), NDArrayIndex.point(10)].dup()

        // forward model
        val resultMotion = modelMotion?.classify(inputMotion)
        resultMotion?.let {
            val (updateResultMotion, updateP4, updateP9) = adjustTranslation(
                resultMotion,
                shoulderTurnRaw,
                shoulderTiltRaw,
                shoulderBendRaw,
                hipTurnRaw,
                hipTiltRaw,
                hipBendRaw,
                isSideView
            )
            return Triple(updateResultMotion, updateP4, updateP9)
        }

        // adjust translation
        return Triple(null, null, null)
    }

    private fun adjustTranslation(resultMotion: INDArray,
                                  sTurnRaw: INDArray,
                                  sTiltRaw: INDArray,
                                  sBendRaw: INDArray,
                                  hTurnRaw: INDArray,
                                  hTiltRaw: INDArray,
                                  hBendRaw: INDArray,
                                  isSideView: Boolean, magicTurnFactor: Double = 1.0): Triple<INDArray, Int, Int> {
        var updatedP4: Int = -1
        var updatedP9: Int = -1
        val jointTranslationIdxList = arrayListOf<Int>(7, 8, 6, 1, 2, 0)
        var tmpResultMotion = resultMotion.dup()
        for (motionIdx in 0 until 12) {
            //---- adjust hip and shoulder
            if (motionIdx == 3) { //shoulderTiltRaw
                tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                    .assign(
                        resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                            .add(sTiltRaw).div(2.0)
                    )
            } else if (motionIdx == 4) { //shoulderBendRaw
                tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                    .assign(
                        resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                            .add(sBendRaw).div(2.0)
                    )
            } else if ( motionIdx == 5) { //shoulderTurnRaw
                if (isSideView) {
                    tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                        .assign(
                            resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                                .add(sTurnRaw.mul(magicTurnFactor)).div(2.0)
                        )
                } else {
                    tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                        .assign(
                            resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                                .add(sTurnRaw).div(2.0)
                        )
                }
            } else if (motionIdx == 9) { //hipTiltRaw
                tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                    .assign(
                        resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                            .add(hTiltRaw).div(2.0)
                    )
            } else if (motionIdx == 10) { //hipBendRaw
                tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                    .assign(
                        resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                            .add(hBendRaw).div(2.0)
                    )
            } else if (motionIdx == 11) { //hipTurnRaw
                tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                    .assign(
                        resultMotion[NDArrayIndex.all(), NDArrayIndex.point(motionIdx.toLong())]
                            .add(hTurnRaw).div(2.0)
                    )
            }

            //========= ToDo Adjust Other Translation
            //========================================//
        }

        //=== Update P4 and P9 Following shoulder turn
        var updateShoulderTurn = tmpResultMotion[NDArrayIndex.all(), NDArrayIndex.point(5)]
        updatedP9 = updateShoulderTurn.toDoubleVector().withIndex().maxBy { it.value }.index
        if (updatedP9 > 0) {
            updatedP4 = updateShoulderTurn[NDArrayIndex.interval(0, updatedP9)].toDoubleVector().withIndex().minBy { it.value }.index
        }
        return Triple(tmpResultMotion, updatedP4, updatedP9)
    }

    private fun processKinematics(localizedPose3D: INDArray, isLefty: Boolean,
                          oHipZeroFrame: INDArray, oShoulderZeroFrame: INDArray): INDArray {

        val rShoulder = localizedPose3D[NDArrayIndex.interval(36, 39)]
        val lShoulder = localizedPose3D[NDArrayIndex.interval(39, 42)]
        val rHip = localizedPose3D[NDArrayIndex.interval(6, 9)]
        val lHip = localizedPose3D[NDArrayIndex.interval(9, 12)]
        val oShoulder = rShoulder.add(lShoulder).div(2.0)
        val oHip = rHip.add(lHip).div(2.0)

        // z-axis: M-L from Right to Left
        var shoulderZ = lShoulder.sub(rShoulder)

        // y-axis: S-I oriented downward
        var shoulderY = oHip.sub(oShoulder)

        // x-axis: A-P from Back to Front
        var shoulderX =  MotionHelperUtils().crossProduct(shoulderY, shoulderZ)

        // (1) ENSURE ORTHOGONALITY
        shoulderY = MotionHelperUtils().crossProduct(shoulderZ, shoulderX)

        // (2) ENSURE NORMALITY
        shoulderX.divi(shoulderX.norm2(0))
        shoulderY.divi(shoulderY.norm2(0))
        shoulderZ.divi(shoulderZ.norm2(0))

        var shoulderLCS = Nd4j.zeros(3, 3)
        shoulderLCS[NDArrayIndex.point(0)].assign(shoulderX)
        shoulderLCS[NDArrayIndex.point(1)].assign(shoulderY)
        shoulderLCS[NDArrayIndex.point(2)].assign(shoulderZ)

        // z-axis: M-L from Right to Left
        var hipZ = lHip.sub(rHip)

        // y-axis: S-I oriented downward
        var hipY = oHip.sub(oShoulder)

        // x-axis: A-P from Back to Front
        var hipX = MotionHelperUtils().crossProduct(hipY, hipZ)

        // (1) ENSURE ORTHOGONALITY
        hipY = MotionHelperUtils().crossProduct(hipZ, hipX)

        // (2) ENSURE NORMALITY
        hipX.divi(hipX.norm2(0))
        hipY.divi(hipY.norm2(0))
        hipZ.divi(hipZ.norm2(0))

        var hipLCS = Nd4j.zeros(3, 3)
        hipLCS[NDArrayIndex.point(0)].assign(hipX)
        hipLCS[NDArrayIndex.point(1)].assign(hipY)
        hipLCS[NDArrayIndex.point(2)].assign(hipZ)

        // CONVERSION FACTORS
        val  meters2inches: Double = 39.37008
        val radians2degrees: Double =  180.0 / Math.PI

        val hipFeature = swingParameters(hipLCS, oHip, oHipZeroFrame, isLefty)
        val thrustHip = hipFeature[0] * meters2inches
        val swayHip = hipFeature[1] * meters2inches
        val liftHip = hipFeature[2] * meters2inches
        val tiltHip = hipFeature[3] * radians2degrees
        val bendHip = hipFeature[4] * radians2degrees
        val turnHip = hipFeature[5] * radians2degrees

        val shoulderFeature = swingParameters(shoulderLCS, oShoulder, oShoulderZeroFrame, isLefty)
        val thrustShoulder = shoulderFeature[0] * meters2inches
        val swayShoulder = shoulderFeature[1] * meters2inches
        val liftShoulder = shoulderFeature[2] * meters2inches
        val tiltShoulder = shoulderFeature[3] * radians2degrees
        val bendShoulder = shoulderFeature[4] * radians2degrees
        val turnShoulder = shoulderFeature[5] * radians2degrees

        val outputFeature = arrayListOf<Double>(thrustShoulder, swayShoulder, liftShoulder, tiltShoulder, bendShoulder, turnShoulder,
            thrustHip, swayHip, liftHip, tiltHip, bendHip, turnHip)

        return Nd4j.create(outputFeature)
    }

    private fun swingParameters(
        DCM: INDArray,
        Origin: INDArray, OriginZeroFrame: INDArray,
        isLefty: Boolean
    ): List<Double> {
        val dcm0Rot = MotionHelperUtils().eye(3, 3)
        val dcmStatic = MotionHelperUtils().eye(3, 3)

        val translation = Origin.sub(OriginZeroFrame).toDoubleVector() // 1 x 3
        val rotation = dcmStatic.mmul(dcm0Rot.mmul(DCM)).toDoubleMatrix() // 1 x 3

        val turn = if (isLefty) {
            atan2(rotation[0][2], (-1.0) * rotation[0][0])
        } else {
            atan2((-1.0) * rotation[0][2], rotation[0][0]) * (-1.0)
        }
        val bend = asin(rotation[0][1])
        val tilt = atan2((-1.0) * rotation[2][1], rotation[1][1]) * (-1.0)
        val thrust = if (isLefty) {
            translation[0] * (-1.0)
        } else {
            translation[0]
        }
        val sway = translation[2] * (-1.0)
        val lift = translation[1]

        return listOf(thrust, sway, lift, tilt, bend, turn)
    }

    private fun extractDesiredJoints(normedPose3D: INDArray, removeJointList: List<Int> = listOf<Int>(10, 11, 14, 15)): INDArray {
        var selectedList = mutableListOf<Long>()
        var selectedNormedPose3D = Nd4j.zeros(totalFrame!!, 12, 3)
        for (jointIdx in 0 until 16) {
            if (!removeJointList.contains(jointIdx)) {
                selectedList.add(jointIdx.toLong())
            }
        }
        selectedNormedPose3D.assign(
            normedPose3D[NDArrayIndex.all(), NDArrayIndex.indices(*selectedList.toLongArray()), NDArrayIndex.all()]
        )
        return selectedNormedPose3D.reshape(totalFrame!!.toLong(), 12 * 3)
    }

    private fun moveStaticPointToCenterHip(uplift3D: INDArray, frameIdxChosen: Int = 0): INDArray {
        val centerHip = uplift3D[NDArrayIndex.point(frameIdxChosen.toLong()), NDArrayIndex.point(UpLift().centerHip.toLong()), NDArrayIndex.all()]
        return  uplift3D.sub(centerHip)
    }
}