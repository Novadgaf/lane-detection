package io.ullmer.lanedetection;

import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2HLS;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2Lab;

import android.util.Pair;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LanePipeline {
    private final Mat originalImage;

    private Mat cameraMtx;
    private Mat cameraDist;

    private Mat M;
    private Mat Minv;

    private int height;
    private int width;

    public LanePipeline(Mat inputImage) {
        this.originalImage = inputImage;
        //calibrateCamera();
    }

    public Mat runPipeline() {
        this.width = this.originalImage.width();
        this.height = this.originalImage.height();

        // TODO: uncomment this to use the camera calibration
        //Mat undistorted = undistort(originalImage);
        Mat warped = warp(originalImage);

        return warped;
//        Mat filtered = filter(warped);
//
//        Pair<WeightedObservedPoints, WeightedObservedPoints> lanePoints = findLanePoints(filtered);
//        double[] leftCoefficients = fitPolynomial(lanePoints.first);
//        double[] rightCoefficients = fitPolynomial(lanePoints.second);
//
//        Mat lanes = drawLanes(filtered.height(), filtered.width(), leftCoefficients, rightCoefficients);
//        Mat unwarped = unwarp(lanes);
//
//        Mat combined = overlay(originalImage, unwarped);
//        double krum = getKruemmung(leftCoefficients, rightCoefficients);
//
//        Mat finalImage = new Mat();
//        Imgproc.putText(combined,
//                "Radius: "+ krum,
//                new Point(50, 50),
//                Imgproc.FONT_HERSHEY_TRIPLEX,
//                1,
//                new Scalar(255, 255, 255));
//
//
//        return finalImage;
    }

    private void calibrateCamera() {
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.001);

        // prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        MatOfPoint3f obj = new MatOfPoint3f();
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 9; j++) {
                obj.push_back(new MatOfPoint3f(new Point3(i, j, 0)));
            }
        }

        // Arrays to store object points and image points from all the images.
        List<MatOfPoint3f> objPoints = new ArrayList<>();
        List<MatOfPoint2f> imgPoints = new ArrayList<>();

        // Step through the list and search for chessboard corners

    }

    private Mat undistort(Mat input) {
        Mat output = new Mat();
        Calib3d.undistort(input, output, this.cameraMtx, this.cameraDist, this.cameraMtx);
        return output;
    }

    private Mat warp(Mat input) {
        Size size = input.size();
        int offset = 200;

        // create original points matrix
        int[] src = { 701, 459,
                      1055, 680,
                      265, 680,
                      580, 459 };
        Mat originalMatrix = new Mat(4, 2, CvType.CV_32S );
        originalMatrix.put(4, 2, src );

        // create destination points matrix
        int[] dest = { this.width - offset, 0,
                        this.width - offset, this.height,
                        offset, this.height,
                        offset, 0 };
        Mat destinationMatrix = new Mat(4, 2, CvType.CV_32S );
        originalMatrix.put(4, 2, dest );

        this.M = Imgproc.getPerspectiveTransform(originalMatrix, destinationMatrix);
        this.Minv = Imgproc.getPerspectiveTransform(destinationMatrix, originalMatrix);

        // warp perspective
        Mat warped = new Mat();
        Imgproc.warpPerspective(input, warped, this.M, size);
        return warped;
    }

    private Mat filter(Mat input) {
        return input;
    }

    private Pair<WeightedObservedPoints, WeightedObservedPoints> findLanePoints(Mat input) {
        final WeightedObservedPoints leftPoints = new WeightedObservedPoints();
        final WeightedObservedPoints rightPoints = new WeightedObservedPoints();

        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); i++) {
                double point = input.get(i, j)[0];
                if (point > 0) {
                    if (j <= this.width/2) {
                        leftPoints.add(i, j);
                    } else {
                        rightPoints.add(i, j);
                    }
                }
            }
        }

        return new Pair<>(leftPoints, rightPoints);
    }

    private double[] fitPolynomial(WeightedObservedPoints points) {
        final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2);

        // Retrieve fitted parameters (coefficients of the polynomial function).
        return fitter.fit(points.toList());
    }

    private Mat drawLanes(int height, int width, double[] wLeft, double[] wRight) {
        Mat image = new Mat();
        PolynomialFunction leftFunction = new PolynomialFunction(wLeft);
        PolynomialFunction rightFunction = new PolynomialFunction(wRight);

        List<MatOfPoint> leftPoints = new ArrayList<>();
        List<MatOfPoint> rightPoints = new ArrayList<>();

        for (int i = 1; i <= this.height; i++ ) {
            double leftY = leftFunction.value(i);
            double rightY = rightFunction.value(i);
            leftPoints.add(new MatOfPoint(new Point(i, leftY)));
            rightPoints.add(new MatOfPoint(new Point(i, rightY)));
        }

        Imgproc.polylines(image, null, false, new Scalar(18, 102, 226), 15);
        Imgproc.polylines(image, null, false, new Scalar(18, 102, 226), 15);

        // rightPoints gets reversed and added to leftPoints. leftPoints contains now all Points
        Collections.reverse(rightPoints);
        leftPoints.addAll(rightPoints);

        Imgproc.fillPoly(image, leftPoints, new Scalar(0, 255, 0));
        return image;
    }

    private double getRadius(double[] parameters) {
        // TODO height oder width??
        return Math.pow(1 + (2 * parameters[0] * this.height + Math.pow(parameters[1], 2)), 1.5) / Math.abs(2*parameters[0]);
    }

    private double getKruemmung(double[] wLeft, double[] wRight) {
        double left = getRadius(wLeft);
        double right = getRadius(wRight);
        return (left + right) / 2;
    }

    private Mat unwarp(Mat input) {
        Mat output = new Mat();
        Imgproc.warpPerspective(input, output, this.Minv, input.size());
        return output;
    }

    private Mat overlay(Mat original, Mat overlay) {
        double opacity = 0.5f;

        Mat result = new Mat();
        org.opencv.core.Core.addWeighted(original, 1.0, overlay, opacity, 0, result);
        return result;
    }

    private Mat[] convert2Lab(Mat input) {
        Mat labImage = new Mat();
        Imgproc.cvtColor(input, labImage, Imgproc.COLOR_BGR2Lab);

        Mat l = new Mat();
        Mat a = new Mat();
        Mat b = new Mat();
        Core.extractChannel(labImage, l, 0);
        Core.extractChannel(labImage, a, 1);
        Core.extractChannel(labImage, b, 2);

        return new Mat[] { l, a, b };
    }

    private Mat[] convert2Hls(Mat input) {
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(input, hsvImage, Imgproc.COLOR_BGR2HLS);
        // Extract the hue, saturation, and value (lightness) channels
        Mat hue = new Mat();
        Mat saturation = new Mat();
        Mat lightness = new Mat();
        Core.extractChannel(hsvImage, hue, 0);
        Core.extractChannel(hsvImage, saturation, 1);
        Core.extractChannel(hsvImage, lightness, 2);

        return new Mat[] { hue, saturation, lightness };
    }

    private Mat normalizeHlsImage(Mat image) {
        Mat hls_l = convert2Hls(image)[1];
        double maxValue = Core.minMaxLoc(hls_l).maxVal;
        Core.multiply(hls_l, new Scalar(255 / maxValue), hls_l);

        Mat zeros = new Mat(hls_l.rows(), hls_l.cols(), hls_l.type());

        // Create the binary mask using the thresholds
        Mat mask = new Mat();
        Core.inRange(hls_l, new Scalar(220), new Scalar(255), mask);

        // TODO: This does not work!
        // Apply the mask to the output array
        // zeros.setTo(new Scalar(0), mask);
        // zeros.setTo(new Scalar(1), mask.not());

        return zeros;
    }
}
