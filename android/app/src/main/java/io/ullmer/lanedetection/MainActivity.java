package io.ullmer.lanedetection;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private io.ullmer.lanedetection.databinding.MainActivityBinding binding;
    private Uri inputUri;

    BaseLoaderCallback baseLoaderCallback= new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {

            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("MainActivity", "onManagerConnected: OpenCV loaded");
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (OpenCVLoader.initDebug()){
            // if loaded successfully
            Log.d("MainActivity", "onResume: OpenCV initialized");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else {
            Log.d("MainActivity", "onResume: OpenCV not initialized");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }

        binding = io.ullmer.lanedetection.databinding.MainActivityBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.buttonRun.setOnClickListener(view -> {
            Bitmap bild = uriToBitmap(inputUri);
            runPipeline(bild);
        });

        binding.inputImage.setOnClickListener(view -> {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), 200);
        });
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == 200) {
                Uri selectedImageUri = data.getData();
                if (null != selectedImageUri) {
                    this.inputUri = selectedImageUri;
                    binding.inputImage.setImageURI(selectedImageUri);
                }
            }
        }
    }

    private void runPipeline(Bitmap bild) {
        Bitmap inputBitmap = bild.copy(Bitmap.Config.ARGB_8888, true);

        Mat input = new Mat();
        Utils.bitmapToMat(inputBitmap, input);

        LanePipeline pipeline = new LanePipeline(input);

        long startTime = System.currentTimeMillis();
        Mat output = pipeline.runPipeline();
        long difference = System.currentTimeMillis() - startTime;

        Bitmap outputBitmap = Bitmap.createBitmap(output.rows(), output.cols(),
                            Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(output, outputBitmap);

        binding.outputImage.setImageBitmap(outputBitmap);

        Toast.makeText(
                        getApplicationContext(),
                        "Pipeline ist gelaufen in " + difference +  "ms",
                        Toast.LENGTH_SHORT)
                .show();
    }

    private Bitmap uriToBitmap(Uri imageUri) {
        try {
            return MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
        } catch (IOException e) {
            Toast.makeText(getApplicationContext(), "File not found", Toast.LENGTH_LONG).show();
        }
        return null;
    }
}