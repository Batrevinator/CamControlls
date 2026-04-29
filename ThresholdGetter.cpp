#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    std::cout << "Threshold Getter - Motion Area Labeling" << std::endl;
    std::cout << "=======================================" << std::endl;

    VideoCapture cap;
    bool cameraFound = false;
    int cameraIndex = -1;
    for (int i = 0; i < 10; ++i) {
        cap.open(i);
        if (cap.isOpened()) {
            cameraIndex = i;
            cameraFound = true;
            std::cout << "Camera found at index " << i << std::endl;
            break;
        }
        cap.release();
    }
    if (!cameraFound) {
        cerr << "Error: No camera found!" << std::endl;
        return -1;
    }
    std::cout << "Using camera index: " << cameraIndex << std::endl;

    Ptr<BackgroundSubtractorMOG2> bgSubtractor = createBackgroundSubtractorMOG2();
    Mat frame, fgMask;
    namedWindow("Motion Area Labeling", WINDOW_AUTOSIZE);
    namedWindow("Blurred Mask", WINDOW_AUTOSIZE);
    namedWindow("Foreground Mask", WINDOW_AUTOSIZE);
    std::cout << "\nControls:\n  'q' or ESC: Quit\n" << std::endl;

    const double MIN_AREA = 1500; // Minimum area threshold for labeling
    const double MAX_AREA = 6500;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not capture frame" << std::endl;
            break;
        }
        bgSubtractor->apply(frame, fgMask);
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
        Mat blurredMask;
        GaussianBlur(fgMask, blurredMask, Size(11, 11), 0);
        threshold(blurredMask, fgMask, 90, 255, THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > MIN_AREA && area < MAX_AREA) {
                Rect bbox = boundingRect(contour);
                rectangle(frame, bbox, Scalar(0,255,0), 2);
                putText(frame, "Area: " + std::to_string((int)area), bbox.tl() + Point(0, -5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2);
            }
        }
        imshow("Motion Area Labeling", frame);
        imshow("Blurred Mask", blurredMask);
        imshow("Foreground Mask", fgMask);
        int key = waitKey(30);
        if (key == 'q' || key == 27) break;
    }
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Motion area labeling stopped." << std::endl;
    return 0;
}
