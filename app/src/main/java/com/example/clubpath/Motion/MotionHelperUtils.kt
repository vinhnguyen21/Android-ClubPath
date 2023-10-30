package com.example.clubpath.Motion

import com.example.clubpath.utils.UpLift
import org.nd4j.linalg.api.ndarray.INDArray
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
}