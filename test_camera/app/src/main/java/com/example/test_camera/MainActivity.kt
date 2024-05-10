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
import android.widget.EditText
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private lateinit var cameraManager: CameraManager
    private lateinit var cameraDevice: CameraDevice
    private lateinit var captureRequestBuilder: CaptureRequest.Builder
    private lateinit var cameraCaptureSession: CameraCaptureSession
    private lateinit var surface: Surface

    private var prevX: Float = 0f
    private var prevY: Float = 0f
    private var isDragging: Boolean = false


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
            showCustomCameraDialog()
        }
    }

    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(
            baseContext, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    private fun showCustomCameraDialog() {
        // Inflar el diseño del diálogo personalizado
        val dialogView = layoutInflater.inflate(R.layout.custom_camera_dialog, null)
        val backgroundEditText: EditText = dialogView.findViewById(R.id.backgroundEditText)
        val cameraPreviewSurfaceView: SurfaceView =
            dialogView.findViewById(R.id.cameraPreviewSurfaceView)

        // Crear el diálogo personalizado
        val builder = AlertDialog.Builder(this)
            .setView(dialogView)
            .setCancelable(true)

        // Crear el diálogo
        val dialog = builder.create()

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
                startCameraPreview(surface)
            }
        })

        // Mostrar el diálogo
        dialog.show()
    }

    private fun startCameraPreview(surface: Surface) {
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        // Obtener la lista de ID de las cámaras disponibles
        val cameraIds = cameraManager.cameraIdList

        // Buscar el ID de la cámara frontal
        var frontCameraId: String? = null
        for (cameraId in cameraIds) {
            val cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId)
            val facing = cameraCharacteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
                frontCameraId = cameraId
                break
            }
        }

        // Verificar si se encontró la cámara frontal
        if (frontCameraId == null) {
            // Manejar la situación en la que no se encuentra la cámara frontal
            return
        }

        // Abrir la cámara frontal
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // Manejar el caso en el que el permiso de la cámara no está concedido
            return
        }
        cameraManager.openCamera(frontCameraId, object : CameraDevice.StateCallback() {
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