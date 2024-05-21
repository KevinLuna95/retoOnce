/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.TextView
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.google.mediapipe.examples.handlandmarker.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.examples.handlandmarker.HandLandmarkerHelper

class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding
    private val viewModel : MainViewModel by viewModels()
    private lateinit var textViewRes: TextView
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var updateRunnable: Runnable

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)
        val res = HandLandmarkerHelper.result_output
        textViewRes = findViewById(R.id.textView_res)

        // Inicializa el Runnable para actualizar el TextView
        updateRunnable = object : Runnable {
            override fun run() {
                Log.d("FUNCIONA", "$HandLandmarkerHelper.result_output")
                // Actualiza el TextView con el valor actual de result_output
                textViewRes.text = HandLandmarkerHelper.result_output
                // Vuelve a ejecutar el Runnable después de 1 segundo
                handler.postDelayed(this, 1000)
            }
        }

        // Comienza la ejecución del Runnable
        handler.post(updateRunnable)
    }
    override fun onDestroy() {
        super.onDestroy()
        // Detiene la ejecución del Runnable cuando la actividad se destruye
        handler.removeCallbacks(updateRunnable)
    }
    override fun onBackPressed() {
       finish()
    }
}