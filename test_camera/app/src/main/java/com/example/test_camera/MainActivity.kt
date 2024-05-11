package com.example.test_camera

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton

class MainActivity : AppCompatActivity() {

    private lateinit var cameraManager: CameraManager
    private lateinit var cameraDevice: CameraDevice
    private lateinit var captureRequestBuilder: CaptureRequest.Builder
    private lateinit var cameraCaptureSession: CameraCaptureSession
    private lateinit var surface: Surface
    private var cameraId: String? = null
    private var frontCameraId: String? = null
    private var backCameraId: String? = null
    private lateinit var switchCameraButton: FloatingActionButton
    private lateinit var cameraPreviewSurfaceView: SurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Solicitar permiso de cámara si no está concedido
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
        } else {
            setupCameraIds()
            setupCustomCameraDialog()
        }
    }

    private fun setupCameraIds() {
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        val cameraIds = cameraManager.cameraIdList
        for (id in cameraIds) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                frontCameraId = id
            } else if (facing == CameraCharacteristics.LENS_FACING_BACK) {
                backCameraId = id
            }
        }
        cameraId = frontCameraId // Por defecto, usar la cámara trasera al iniciar
    }

    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(
            baseContext, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    private fun setupCustomCameraDialog() {
        cameraPreviewSurfaceView = findViewById(R.id.cameraPreviewSurfaceView)
        switchCameraButton = findViewById(R.id.switchCameraButton)

        // Configurar el SurfaceView para mostrar la vista previa de la cámara
        val surfaceHolder = cameraPreviewSurfaceView.holder
        surfaceHolder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) {
                // Aquí puedes realizar acciones cuando la superficie cambia, como actualizar la configuración de la cámara
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                // Liberar recursos relacionados con la cámara cuando la superficie se destruye
                cameraDevice.close()
            }

            override fun surfaceCreated(holder: SurfaceHolder) {
                // Configurar la vista previa de la cámara cuando la superficie se crea
                surface = holder.surface
                startCameraPreview()
            }
        })

        // Manejar clics en el botón de cambio de cámara
        switchCameraButton.setOnClickListener {
            if (cameraId == backCameraId) {
                cameraId = frontCameraId // Cambiar a la cámara frontal
            } else {
                cameraId = backCameraId // Cambiar a la cámara trasera
            }
            startCameraPreview()
        }
    }

    private fun startCameraPreview() {
        cameraManager.openCamera(cameraId!!, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                captureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequestBuilder.addTarget(surface)
                cameraDevice.createCaptureSession(
                    listOf(surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            cameraCaptureSession = session
                            captureRequestBuilder.set(
                                CaptureRequest.CONTROL_MODE,
                                CaptureRequest.CONTROL_MODE_AUTO
                            )
                            cameraCaptureSession.setRepeatingRequest(
                                captureRequestBuilder.build(),
                                null,
                                null
                            )
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {
                            // Manejar el caso en el que falla la configuración de la sesión de captura
                        }
                    },
                    null
                )
            }

            override fun onDisconnected(camera: CameraDevice) {
                cameraDevice.close()
            }

            override fun onError(camera: CameraDevice, error: Int) {
                cameraDevice.close()
            }
        }, null)
    }

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 10
    }
}