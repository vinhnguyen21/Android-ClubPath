package com.example.clubpath.Motion

import com.example.clubpath.utils.UpLift
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import kotlin.math.abs

class MotionHelperUtils {
    fun findDeltaAxis(pose3D: INDArray, p1: Int = 75): Double {
        val leftAnkle =
            pose3D[NDArrayIndex.point(p1.toLong()), NDArrayIndex.point(UpLift().leftAnkle.toLong()), NDArrayIndex.all()].dup()
                .div(2.0)
        val rightAnkle =
            pose3D[NDArrayIndex.point(p1.toLong()), NDArrayIndex.point(UpLift().rightAnkle.toLong()), NDArrayIndex.all()].dup()
                .div(2.0)
        val head =
            pose3D[NDArrayIndex.point(p1.toLong()), NDArrayIndex.point(UpLift().head.toLong()), NDArrayIndex.all()].dup()
        val centerAnKleMF = leftAnkle.add(rightAnkle)
        return abs(
            head[NDArrayIndex.point(1)].sub(centerAnKleMF[NDArrayIndex.point(1)])
                .toDoubleVector()[0]
        )
    }

    fun crossProduct(point3DOne: INDArray, point3dTwo: INDArray): INDArray {
        val tmpPointOne = point3DOne.toDoubleVector()
        val tmpPointTwo = point3dTwo.toDoubleVector()
        val resultX = tmpPointOne[1] * tmpPointTwo[2] - tmpPointOne[2] * tmpPointTwo[1]
        val resultY = tmpPointOne[2] * tmpPointTwo[0] - tmpPointOne[0] * tmpPointTwo[2]
        val resultZ = tmpPointOne[0] * tmpPointTwo[1] - tmpPointOne[1] * tmpPointTwo[0]

        val output = arrayListOf<Double>(resultX, resultY, resultZ)
        return Nd4j.create(output)
    }

    fun eye(rows: Int, cols: Int): INDArray {
        var output = Nd4j.zeros(rows, cols)
        val minSize = if (rows < cols) rows else cols
        for (i in 0 until minSize) {
            output[NDArrayIndex.point(i.toLong()), NDArrayIndex.point(i.toLong())].assign(1.0)
        }
        return output
    }

    fun smoothKpts(kptArray: INDArray, windowSize: Int): INDArray? {
        val shapeArray = kptArray.shape()
        val numJoint: Int = shapeArray[1].toInt()
        val numCoordinate:Int = shapeArray[2].toInt()

        val smoothKptOut = Nd4j.zeros(*shapeArray)

        for (jointIdx in 0 until numJoint) {
            for (axisIdx in 0 until numCoordinate) {
                val tmpArrayJoint = kptArray[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(axisIdx.toLong())].toDoubleVector()
                smoothKptOut[NDArrayIndex.all(), NDArrayIndex.point(jointIdx.toLong()), NDArrayIndex.point(axisIdx.toLong())]
                    .assign(arrConv(tmpArrayJoint, windowSize))
            }
        }
        return smoothKptOut
    }

    private fun arrConv(rawArray: DoubleArray, windowSize: Int): INDArray {
        val filter = DoubleArray(windowSize) {1.0 / windowSize}
        val smoothResult = vDSP_convD(rawArray, filter)

        // Create a padded input array with nearest input extension
        val extensionSize: Int = windowSize / 2
        val paddedOutput = DoubleArray(rawArray.count()) {0.0}

        for (i in 0 until extensionSize) {
            paddedOutput[i] = smoothResult.first()
        }
        // Copy the original input data
        for (i in 0 until smoothResult.count())  {
            paddedOutput[i + extensionSize] = smoothResult[i]
        }
        // Extend on the right side with nearest value
        for (i in smoothResult.count() + extensionSize until rawArray.count()) {
            paddedOutput[i] = smoothResult.last()
        }

        return Nd4j.create(paddedOutput)
    }

    fun vDSP_convD(signal: DoubleArray, filter: DoubleArray): DoubleArray {
//        val signal = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8)
//        val filter = intArrayOf(10, 20, 30)
        val outputCount = signal.size - filter.size + 1
        val conv = mutableListOf<Double>()
        for(i in 0 until outputCount){
            var result = 0.0
            for(x in filter.indices){
                result += (signal[i + x] * filter[x])
            }
            conv.add(result)
        }
        return conv.toDoubleArray()
    }

}