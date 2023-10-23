package com.example.clubpath.utils

import android.content.Context
import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader

@Serializable
data class SwingKeypointModel(var bboxList: List<Double>,
                            var listPose: List<KeyPose>,
                            var frame: Int,
                            var time: Int,
                            var bboxBallDraw: List<Double>,
                            var bboxClubDraw: List<Double>,
                            var bboxHeadDraw: List<Double>,
                            var optHeadBox: List<Double>,
                            var bboxClub: List<List<Double>>,
                            var bboxBall: List<List<Double>>,
                            var bboxHead: List<List<Double>>)

@Serializable
data class KeyPose(var landmarks: List<KeyLandmark>)

@Serializable
data class KeyLandmark(var name: String,
                       var location: KeyPoint)

@Serializable
data class KeyPoint(var x: Double,
                    var y: Double,
                    var prob: Double)

fun readJSONFromAssets(context: Context, path: String): List<SwingKeypointModel> {
    val identifier = "[ReadJSON]"
    try {
        val inputStream = context.assets.open(path)
        // Read the JSON file into a String
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        return Json.decodeFromString<List<SwingKeypointModel>>(jsonString)
    } catch (e: Exception) {
        Log.e(
            identifier,
            "Error reading JSON: $e.",
        )
        e.printStackTrace()
        return emptyList()
    }
}