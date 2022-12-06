package io.ullmer.lanedetection;

import org.opencv.core.Mat;

public class LanePipeline {
    private Mat inputImage;

    public LanePipeline(Mat inputImage) {
        this.inputImage = inputImage;
    }

    public Mat runPipeline() {
        return inputImage;
    }
}
