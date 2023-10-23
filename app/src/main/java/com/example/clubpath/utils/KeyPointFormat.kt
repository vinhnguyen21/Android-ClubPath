package com.example.clubpath.utils

data class CoCoFormat(
    val nose: Int = 0,
    val leftEye: Int = 1,
    val rightEye: Int = 2,
    val leftEar: Int = 3,
    val rightEar: Int = 4,
    val leftShoulder: Int = 5,
    val rightShoulder: Int = 6,
    val leftElbow: Int = 7,
    val rightElbow: Int = 8,
    val leftWrist: Int = 9,
    val rightWrist: Int = 10,
    val leftHip: Int = 11,
    val rightHip: Int = 12,
    val leftKnee: Int = 13,
    val rightKnee: Int = 14,
    val leftAnkle: Int = 15,
    val rightAnkle: Int = 16
)

data class Human36M(
    val pelvis: Int = 0,
    val rightHip: Int = 1,
    val rightKnee: Int = 2,
    val rightFoot: Int = 3,
    val leftHip: Int = 4,
    val leftKnee: Int = 5,
    val leftFoot: Int = 6,
    val spine: Int = 7,
    val thorax: Int = 8,
    val neckBase: Int = 9,
    val head: Int = 10,
    val leftShoulder: Int = 11,
    val leftElbow: Int = 12,
    val leftWrist: Int = 13,
    val rightShoulder: Int = 14,
    val rightElbow: Int = 15,
    val rightWrist: Int = 16
)

data class UpLift(
    val rightAnkle: Int = 0,
    val rightKnee: Int = 1,
    val rightHip: Int = 2,
    val leftHip: Int = 3,
    val leftKnee: Int = 4,
    val leftAnkle: Int = 5,
    val centerHip: Int = 6,
    val centerShoulder: Int = 7,
    val neck: Int = 8,
    val head: Int = 9,
    val rightWrist: Int = 10,
    val rightElbow: Int = 11,
    val rightShoulder: Int = 12,
    val leftShoulder: Int = 13,
    val leftElbow: Int = 14,
    val leftWrist: Int = 15
)

data class HumanMotion(
    val shoulderThrust: Int = 0,
    val shoulderSway: Int = 1,
    val shoulderLift: Int = 2,
    val shoulderTilt: Int = 3,
    val shoulderBend: Int = 4,
    val shoulderTurn: Int = 5,
    val hipThrust: Int = 6,
    val hipSway: Int = 7,
    val hipLift: Int = 8,
    val hipTilt: Int = 9,
    val hipBend: Int = 10,
    val hipTurn: Int = 11,
    val trailingKneeAngle: Int = 12,
    val leadingElbowAngle: Int = 13,
    val leadingShoulderAdduction: Int = 14,
    val elbowSpan: Int = 15
)