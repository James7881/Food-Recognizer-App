package com.jamsher.foodanalyzer

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private val IMAGE_PICK_CODE = 1000
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var uploadButton: Button
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView    = findViewById(R.id.imageView)
        resultText   = findViewById(R.id.resultText)
        uploadButton = findViewById(R.id.uploadButton)

        // Load the quantized MobileNetV2 model and labels
        interpreter = Interpreter(loadModelFile("mobilenet_v2_1.0_224_quant.tflite"))
        labels      = loadLabels("labels.txt")

        uploadButton.setOnClickListener {
            Intent(Intent.ACTION_PICK).also { intent ->
                intent.type = "image/*"
                startActivityForResult(intent, IMAGE_PICK_CODE)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == IMAGE_PICK_CODE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                contentResolver.openInputStream(uri)?.use { stream ->
                    val originalBitmap = BitmapFactory.decodeStream(stream)
                    val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, 224, 224, true)
                    imageView.setImageBitmap(resizedBitmap)
                    classifyImage(resizedBitmap)
                }
            }
        }
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        assets.openFd(filename).apply {
            FileInputStream(fileDescriptor).channel.use { fc ->
                return fc.map(
                    FileChannel.MapMode.READ_ONLY,
                    startOffset,
                    declaredLength
                )
            }
        }
    }

    private fun loadLabels(filename: String): List<String> =
        assets.open(filename).bufferedReader().readLines()

    private fun classifyImage(bitmap: Bitmap) {
        // Prepare input buffer (UINT8)
        val inputBuffer = convertBitmapToByteBuffer(bitmap)

        // Prepare output array (quantized UINT8 scores)
        val output = Array(1) { ByteArray(labels.size) }

        // Run inference
        interpreter.run(inputBuffer, output)

        // Find the index with highest score
        val scores   = output[0]
        val topIndex = scores.indices.maxByOrNull { scores[it].toInt() and 0xFF } ?: -1

        // Convert to human-readable
        val label      = labels.getOrNull(topIndex) ?: "Unknown"
        val rawScore   = scores[topIndex].toInt() and 0xFF
        val confidence = rawScore * 100f / 255f

        resultText.text = "Prediction: $label (%.2f%%)".format(confidence)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(224 * 224 * 3) // UINT8
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        bitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        for (pixel in intValues) {
            byteBuffer.put(((pixel shr 16) and 0xFF).toByte()) // R
            byteBuffer.put(((pixel shr 8)  and 0xFF).toByte()) // G
            byteBuffer.put(( pixel        and 0xFF).toByte()) // B
        }
        return byteBuffer
    }
}
