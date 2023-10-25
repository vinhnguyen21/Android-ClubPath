package com.example.clubpath.components3D

import com.example.clubpath.utils.CoCoFormat
import com.example.clubpath.utils.Human36M
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Utils3DHelper {
    fun isSideViewCheck(keypoint: INDArray, considerFramed: Int = 10): Boolean {
        val xHead = keypoint.getDouble(considerFramed, Human36M().head, 0)
        val xRightShoulder = keypoint.getDouble(considerFramed, Human36M().rightShoulder, 0)
        val xLeftShoulder = keypoint.getDouble(considerFramed, Human36M().leftShoulder, 0)

        if (xRightShoulder < xHead &&  xHead < xLeftShoulder) { return false }
        return true
    }

    fun isLeftyCheck(keypoint: INDArray, isSideView: Boolean, numFrameCheck: Int): Boolean {
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

    fun convertCoCoToHuman36M(cocoArray: INDArray, totalFrame: Int, frameWidth: Double, frameHeight: Double): Pair<INDArray?, INDArray?> {
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

    fun normalizeScreenCoordinate(h36mArray: INDArray,
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

    fun predict3D(rawKptHuman36m: INDArray, processedHuman36M: INDArray,
                  isSideView: Boolean, isLefty: Boolean,
                  frameWidth: Double, frameHeight: Double): Pair<INDArray?, INDArray?> {
        val totalFrame: Int = rawKptHuman36m.shape()[0].toInt()
        val upLift2D = mappingUpliftOrder(rawKptHuman36m, totalFrame)

        for (frameIndex in 0 until totalFrame) {
            val poseSequence2D = extractPoseSequence(processedHuman36M, frameIndex)
        }
        return Pair(null, null)
    }

    private fun mappingUpliftOrder(h36mArray: INDArray, totalFrame: Int): INDArray {
        val keypointShape = IntArray(3)
        keypointShape[0] = totalFrame
        keypointShape[1] = 16
        keypointShape[2] = 2

        val upliftOrder = intArrayOf(3,2,1,4,5,6,0,8,10,9,16,15,14,11,12,13)
        var upliftKpt = Nd4j.zeros(keypointShape, DataType.DOUBLE)

        upliftOrder.forEachIndexed { normalIdx, upliftMapIdx ->
            val tmpJointH36M = h36mArray.get(NDArrayIndex.all(), NDArrayIndex.point(upliftMapIdx.toLong()), NDArrayIndex.all())
            upliftKpt.get(NDArrayIndex.all(), NDArrayIndex.point(normalIdx.toLong()), NDArrayIndex.all()).assign(tmpJointH36M)
        }
        return upliftKpt
    }

    private fun extractPoseSequence(keypointProcesses: INDArray, currentFrameIdx: Int, tempFrameLength: Int = 27): INDArray {
        var input2D = Nd4j.zeros(tempFrameLength, 17, 2)
        var sequenceInput = Nd4j.zeros(tempFrameLength, 16, 2)
        var desiredJoint = intArrayOf(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16)
        val frameLeft: Int = (tempFrameLength - 1) / 2
        val frameRight = frameLeft
        val numFrames: Int = keypointProcesses.shape()[0].toInt()

        val padLeft = maxOf(0, frameLeft - currentFrameIdx)
        val padRight = maxOf(0, frameRight - (numFrames - 1 - currentFrameIdx))

        val startIdx = maxOf(0, currentFrameIdx - frameLeft)
        val endIdx = minOf(numFrames, currentFrameIdx + frameRight + 1)

        // retrieve sequence frames
        if (padLeft != 0) {
            for (leftIdx in 0 until padLeft) {
                input2D.get(NDArrayIndex.point(leftIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all())
                    .assign(keypointProcesses.getRow(0)) //keypointProcesses[0, Matft.all, Matft.all]
            }

            for (tmpIdx in startIdx until  endIdx) {
                input2D.get(NDArrayIndex.point((padLeft + tmpIdx).toLong()), NDArrayIndex.all(), NDArrayIndex.all())
                    .assign(keypointProcesses.getRow(tmpIdx.toLong())) //keypointProcesses[tmpIdx, Matft.all, Matft.all]
            }
        }
        else  {
            if (padRight != 0) {
                for (rightIdx in 0 until padRight) {
                    input2D.get(NDArrayIndex.point((tempFrameLength - rightIdx - 1).toLong()), NDArrayIndex.all(), NDArrayIndex.all())
                        .assign(keypointProcesses.getRow((numFrames - 1).toLong())) //keypointProcesses[-1, Matft.all, Matft.all]
                }
            }
            for ((inputIdx, processIdx) in (0 until (tempFrameLength - padRight)).zip(startIdx until endIdx)) {
                input2D.get(NDArrayIndex.point(inputIdx.toLong()), NDArrayIndex.all(), NDArrayIndex.all())
                    .assign(keypointProcesses.getRow(processIdx.toLong()))
            }
        }

        // remove Nose Joint
        val noseJoint = 9
        var updatedArray = Nd4j.zeros(1, tempFrameLength, 16, 2)
        updatedArray.getRow(0)
            .assign(Nd4j.concat(1,
                input2D.get(NDArrayIndex.interval(0, noseJoint), NDArrayIndex.all(), NDArrayIndex.all()).dup(),
                input2D.get(NDArrayIndex.interval(noseJoint + 1, input2D.shape()[1].toInt()), NDArrayIndex.all(), NDArrayIndex.all()).dup()
            ))
        return updatedArray
    }
}
