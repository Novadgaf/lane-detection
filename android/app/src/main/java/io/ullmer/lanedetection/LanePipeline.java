package io.ullmer.lanedetection;

import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2RGBA;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.cornerSubPix;

import android.content.Context;
import android.util.Pair;

import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.opencv.android.Utils;
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
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LanePipeline {
    private static final DecimalFormat df = new DecimalFormat("0.00");

    private Mat originalImage;

    private final Mat cameraMtx = new Mat();
    private final Mat cameraDist = new Mat();

    private Mat Minv;

    private int height;
    private int width;
    private final Context context;

    /**
     * Initialize the Lane pipeline with a context.
     * Runs the camera calibration
     * @param context the android context (used for toast messages)
     *
     * @throws IOException when calibration images are not found
     */
    public LanePipeline(Context context) throws IOException {
        this.context = context;
        calibrateCamera(new Size(9, 6));
    }

    /**
     * Sets the input image for the pipeline.
     * @param inputImage the input image in OpenCV Mat format
     */
    public void setInputImage(Mat inputImage) {
        this.originalImage = inputImage;
        this.height = inputImage.height();
        this.width = inputImage.width();
    }

    /**
     * Runs the pipeline and returns the output image.
     * @return output image with detected lanes in OpenCV Mat format
     */
    public Mat runPipeline() {
        this.width = this.originalImage.width();
        this.height = this.originalImage.height();

        Mat undistorted = undistort(originalImage);
        Mat warped = warp(undistorted);

        Mat filtered = filter(warped);

        Pair<WeightedObservedPoints, WeightedObservedPoints> lanePoints = findLanePoints(filtered);
        double[] leftCoefficients = fitPolynomial(lanePoints.first);
        double[] rightCoefficients = fitPolynomial(lanePoints.second);

        Mat lanes = drawLanes(filtered.height(), filtered.width(), leftCoefficients, rightCoefficients);
        //return lanes;
        Mat unwarped = unwarp(lanes);

        Imgproc.cvtColor(unwarped, unwarped, COLOR_RGB2RGBA);
        Mat combined = overlay(originalImage, unwarped);
        double krum = getKruemmung(leftCoefficients, rightCoefficients);

        Imgproc.putText(combined,
                "Radius: "+ df.format(krum),
                new Point(50, 50),
                Imgproc.FONT_HERSHEY_TRIPLEX,
                1,
                new Scalar(255, 255, 255));
        
        return combined;
    }

    /**
     * Calibrates the camera using the calibration images in the assets folder.
     * @param chessboardSize the size of the chessboard used for calibration
     * @throws IOException when calibration images are not found
     */
    private void calibrateCamera(Size chessboardSize) throws IOException {
        int[] calibrationImages = {
                R.drawable.calibration1,
                R.drawable.calibration2,
                R.drawable.calibration3,
                R.drawable.calibration4,
                R.drawable.calibration5,
                R.drawable.calibration6,
                R.drawable.calibration7,
                R.drawable.calibration8,
                R.drawable.calibration9,
                R.drawable.calibration10,
                R.drawable.calibration11,
                R.drawable.calibration12,
                R.drawable.calibration13,
                R.drawable.calibration14,
                R.drawable.calibration15,
                R.drawable.calibration16,
                R.drawable.calibration17,
                R.drawable.calibration18,
                R.drawable.calibration19,
                R.drawable.calibration20
        };
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.001);

        // prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        MatOfPoint3f obj = new MatOfPoint3f();
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 9; j++) {
                obj.push_back(new MatOfPoint3f(new Point3(i, j, 0)));
            }
        }

        // Arrays to store object points and image points from all the images.
        List<Mat> objPoints = new ArrayList<>();
        List<Mat> imgPoints = new ArrayList<>();

        Size imageSize = null;

        // Step through the list and search for chessboard corners
        for (int image: calibrationImages) {
            Mat img = Utils.loadResource(this.context, image, Imgcodecs.IMREAD_COLOR);
            imageSize = img.size();
            Imgproc.cvtColor(img, img, COLOR_RGB2GRAY);

            MatOfPoint2f corners = new MatOfPoint2f();
            if (Calib3d.findChessboardCorners(img, chessboardSize, corners)) {
                objPoints.add(obj);
                cornerSubPix(img, corners, new Size(11, 11), new Size(-1, -1), criteria);
                imgPoints.add(corners);
            }
        }

        // init needed variables according to OpenCV docs
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();
        cameraMtx.put(0, 0, 1);
        cameraMtx.put(1, 1, 1);
        Calib3d.calibrateCamera(objPoints, imgPoints, imageSize, this.cameraMtx, this.cameraDist, rvecs, tvecs);
    }

    /**
     * Undistorts the image using the camera calibration.
     * @param input the input image in OpenCV Mat format
     * @return the undistorted image in OpenCV Mat format
     */
    private Mat undistort(Mat input) {
        Mat output = new Mat();
        Calib3d.undistort(input, output, this.cameraMtx, this.cameraDist, this.cameraMtx);
        return output;
    }

    /**
     * warps the image to a top-down view
     * @param input the input image in OpenCV Mat format
     * @return the warped image in OpenCV Mat format
     */
    private Mat warp(Mat input) {
        Size size = input.size();
        int offset = 200;

        // create original points matrix
        float[] src = { 701, 459,
                      1055, 680,
                      265, 680,
                      580, 459 };
        Mat originalMatrix = new Mat(4, 1, CvType.CV_32FC2 );
        originalMatrix.put(0, 0, src );

        // create destination points matrix
        float[] dest = { this.width - offset, 0,
                        this.width - offset, this.height,
                        offset, this.height,
                        offset, 0 };
        Mat destinationMatrix = new Mat(4, 1, CvType.CV_32FC2 );
        destinationMatrix.put(0, 0, dest );

        Mat M = Imgproc.getPerspectiveTransform(originalMatrix, destinationMatrix);
        originalMatrix.checkVector(2, CvType.CV_32F);
        this.Minv = Imgproc.getPerspectiveTransform(destinationMatrix, originalMatrix);

        // warp perspective
        Mat warped = new Mat();
        Imgproc.warpPerspective(input, warped, M, size);
        return warped;
    }

    /**
     * Filters the image to only show the lane lines 1px wide
     * @param input the input image in OpenCV Mat format
     * @return the filtered image in OpenCV Mat format
     */
    private Mat filter(Mat input) {
        Mat hls = convert2Hls(input);
        Mat lab = convert2Lab(input);

        Mat whiteRange = new Mat();
        Mat yellowRange = new Mat();

        Core.inRange(hls, new Scalar(0, 220, 0), new Scalar(255, 255, 255), whiteRange);
        // TODO: wieso ist hier ein anderer Threshold als in Python?
        Core.inRange(lab, new Scalar(0, 0, 150), new Scalar(255, 255, 255), yellowRange);
        // inRange gibt ein Binary Bild zurück

        Mat lines = new Mat();
        Core.add(yellowRange, whiteRange, lines);

        // reduce lines to 1px width
        Mat kernel = new Mat(3, 3, CvType.CV_32S);
        kernel.put(0, 0, -1, 0, 1, -1, 0, 1, -1, 0, 1);
        Mat perwitt = new Mat();
        Imgproc.filter2D(lines, perwitt, -1, kernel);
        Mat output = new Mat();
        Imgproc.threshold(perwitt, output, 0, 255, THRESH_BINARY);
        return output;
    }

    /**
     * Finds the white pixels in the image and returns the x and y coordinates of the pixels
     * This also filters for left and right lane lines
     * @param input the filtered input image in OpenCV Mat format
     * @return first element are the x and y coordinates of the left lane line, second element are the x and y coordinates of the right lane line
     */
    private Pair<WeightedObservedPoints, WeightedObservedPoints> findLanePoints(Mat input) {
        final WeightedObservedPoints leftPoints = new WeightedObservedPoints();
        final WeightedObservedPoints rightPoints = new WeightedObservedPoints();

        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) {
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

    /**
     * fits a 2nd order polynomial on the given points
     * @param points the points to fit the polynomial on
     * @return the coefficients of the polynomial
     */
    private double[] fitPolynomial(WeightedObservedPoints points) {
        final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(2);

        // Retrieve fitted parameters (coefficients of the polynomial function).
        return fitter.fit(points.toList());
    }

    /**
     * Draws lane lines on a blank image
     * @param height the height of the image to draw
     * @param width the width of the image to draw
     * @param wLeft the coefficients of the left lane line
     * @param wRight the coefficients of the right lane line
     * @return the image with the lane lines drawn
     */
    private Mat drawLanes(int height, int width, double[] wLeft, double[] wRight) {
        Mat image = new Mat(height, width, CvType.CV_8UC3);
        PolynomialFunction leftFunction = new PolynomialFunction(wLeft);
        PolynomialFunction rightFunction = new PolynomialFunction(wRight);

        List<Point> leftPoints = new ArrayList<>();
        List<Point> rightPoints = new ArrayList<>();

        for (int i = 0; i < this.height; i++ ) {
            double leftY = leftFunction.value(i);
            double rightY = rightFunction.value(i);
            leftPoints.add(new Point(leftY, i));
            rightPoints.add(new Point(rightY, i));
        }

        List<MatOfPoint> left = new ArrayList<>();
        left.add(new MatOfPoint(leftPoints.toArray(new Point[0])));
        List<MatOfPoint> right = new ArrayList<>();
        right.add(new MatOfPoint(rightPoints.toArray(new Point[0])));

        Imgproc.polylines(image, left, false, new Scalar(18, 102, 226), 15);
        Imgproc.polylines(image, right, false, new Scalar(18, 102, 226), 15);

        Collections.reverse(rightPoints);
        leftPoints.addAll(rightPoints);

        List<MatOfPoint> allPoints = new ArrayList<>();
        allPoints.add(new MatOfPoint(leftPoints.toArray(new Point[0])));

        Imgproc.fillPoly(image, allPoints, new Scalar(0, 255, 0));
        return image;
    }

    /**
     * Calculates the radius of the curvature of a single lane line
     * @param parameters the coefficients of the polynomial
     * @return the radius of the curvature
     */
    private double getRadius(double[] parameters) {
        return Math.pow((1 + Math.pow(2 * parameters[2] * this.width + parameters[1], 2)), 1.5) / Math.abs(2 * parameters[2]);
    }

    /**
     * Calculates the radius of the curvature of the lane
     * @param wLeft the coefficients of the left lane line
     * @param wRight the coefficients of the right lane line
     * @return the radius of the lane
     */
    private double getKruemmung(double[] wLeft, double[] wRight) {
        double left = getRadius(wLeft);
        double right = getRadius(wRight);
        return (left + right) / 2;
    }

    /**
     * Unwarps the image back to the original perspective
     * @param input the input image in OpenCV Mat format
     * @return the unwarped image in OpenCV Mat format
     */
    private Mat unwarp(Mat input) {
        Mat output = new Mat();
        Imgproc.warpPerspective(input, output, this.Minv, input.size());
        return output;
    }

    /**
     * Overlays two images on top of each other.
     * This is used to draw the lane lines on the original image
     * @param original the original image
     * @param overlay the image to overlay on top of the original image
     * @return the original image with the overlay image on top of it
     */
    private Mat overlay(Mat original, Mat overlay) {
        double opacity = 0.5f;

        Mat result = new Mat();
        org.opencv.core.Core.addWeighted(original, 1.0, overlay, opacity, 0, result);
        return result;
    }

    /**
     * Converts the image to Lab color space
     * @param input the input image in OpenCV Mat format in RGB
     * @return the image in Lab color space
     */
    private Mat convert2Lab(Mat input) {
        Mat labImage = new Mat();
        Imgproc.cvtColor(input, labImage, Imgproc.COLOR_RGB2Lab);
        return labImage;
    }

    /**
     * Converts the image to HLS color space
     * @param input the input image in OpenCV Mat format in RGB
     * @return the image in HLS color space
     */
    private Mat convert2Hls(Mat input) {
        Mat hlsImage = new Mat();
        Imgproc.cvtColor(input, hlsImage, Imgproc.COLOR_RGB2HLS);
        return hlsImage;
    }
}
