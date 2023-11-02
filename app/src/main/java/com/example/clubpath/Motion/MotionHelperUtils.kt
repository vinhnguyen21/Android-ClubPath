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

}