package io.ullmer.lanedetection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;

import io.ullmer.lanedetection.databinding.ActivityBeautifulBinding;

public class BeautifulActivity extends AppCompatActivity {

    private ActivityBeautifulBinding binding;
    private Uri inputUri;
    private LanePipeline lanePipeline;
    private String title;
    private Bitmap outputBitmap = null;

    BaseLoaderCallback baseLoaderCallback= new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("MainActivity", "onManagerConnected: OpenCV loaded");
                    try {
                        lanePipeline = new LanePipeline(getApplicationContext());
                    } catch (IOException e) {
                        Log.e("PIEPELINE", "error initializing pipeline: ", e);
                        Toast.makeText(getApplicationContext(), "Error initializing pipeline", Toast.LENGTH_LONG).show();
                    }
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityBeautifulBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        if (OpenCVLoader.initDebug()){
            // if loaded successfully
            Log.d("MainActivity", "onResume: OpenCV initialized");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else {
            Log.d("MainActivity", "onResume: OpenCV not initialized");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }

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

        binding.outputImage.setOnLongClickListener(view -> {
            if (outputBitmap == null) {
                Toast.makeText(getApplicationContext(), "No output image to save", Toast.LENGTH_LONG).show();
                return false;
            }
            shareOutput(outputBitmap);
            return true;
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

        lanePipeline.setInputImage(input);

        Mat output;
        long difference;
        try {
            long startTime = System.currentTimeMillis();
            output = lanePipeline.runPipeline();
            difference = System.currentTimeMillis() - startTime;
        } catch (Exception e) {
            Log.e("PIPELINE", "error running pipeline: ", e);
            Toast.makeText(getApplicationContext(), "Error running pipeline. Please try another image!", Toast.LENGTH_LONG).show();
            return;
        }

        Bitmap outputBitmap = Bitmap.createBitmap(output.width(), output.height(),
                Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(output, outputBitmap);

        this.outputBitmap = outputBitmap;

        binding.outputImage.setImageBitmap(outputBitmap);

        Log.i("PIPELINE", "Pipeline ist gelaufen in " + difference +  "ms");

        Toast.makeText(
                        getApplicationContext(),
                        "Pipeline ist gelaufen in " + difference +  "ms",
                        Toast.LENGTH_SHORT)
                .show();
    }

    private Bitmap uriToBitmap(Uri imageUri) {
        try {
            this.title = imageUri.getLastPathSegment();
            return MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
        } catch (IOException e) {
            Toast.makeText(getApplicationContext(), "File not found", Toast.LENGTH_LONG).show();
        }
        return null;
    }

    private void shareOutput(Bitmap bitmap) {
        String bitmapPath = MediaStore.Images.Media.insertImage(getContentResolver(), bitmap, this.title, "Lane Detection");
        Uri bitmapUri = Uri.parse(bitmapPath);

        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType("image/jpeg");
        intent.putExtra(Intent.EXTRA_STREAM, bitmapUri);
        startActivity(Intent.createChooser(intent, "Share"));
    }
}