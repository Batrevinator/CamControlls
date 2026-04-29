#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

// Helper function to smooth a vector of points using a moving average
std::vector<cv::Point> smoothTrajectory(const std::vector<cv::Point>& trajectory, int windowSize = 5) {
    std::vector<cv::Point> smoothed;
    int n = trajectory.size();
    for (int i = 0; i < n; ++i) {
        int count = 0;
        int sumX = 0, sumY = 0;
        for (int j = std::max(0, i - windowSize/2); j <= std::min(n - 1, i + windowSize/2); ++j) {
            sumX += trajectory[j].x;
            sumY += trajectory[j].y;
            ++count;
        }
        smoothed.push_back(cv::Point(sumX / count, sumY / count));
    }
    return smoothed;
}

int main(int argc, char** argv)
{
    std::cout << "Motion Path Drawing System" << std::endl;
    std::cout << "==========================" << std::endl;

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

    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera properties:" << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;

    Ptr<BackgroundSubtractorMOG2> bgSubtractor = createBackgroundSubtractorMOG2();
    Mat frame, fgMask;
    namedWindow("Motion Path", WINDOW_AUTOSIZE);
    std::cout << "\nControls:\n  'q' or ESC: Quit\n  'c': Clear path\n" << std::endl;
    std::cout << "\nStarting motion path drawing..." << std::endl;

    std::vector<cv::Point> trajectory;
    Mat pathCanvas(height, width, CV_8UC3, Scalar(0,0,0));
    const int MAX_FRAMES = 32;
    const double MAX_AREA = 6500.0;
    const double MIN_AREA = 1500.0;
    cv::Point lastHandPos(-1, -1);
    const double DIST_WEIGHT = 2.0;

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
        GaussianBlur(fgMask, fgMask, Size(11, 11), 0);
        threshold(fgMask, fgMask, 90, 255, THRESH_BINARY);


        std::vector<std::vector<cv::Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> trackedContour;
        double bestScore = -1e9;
        cv::Point bestCentroid(-1, -1);
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            //if (area > MIN_AREA && area < MAX_AREA) {
                // Create a mask for this contour
                Mat contourMask = Mat::zeros(fgMask.size(), CV_8UC1);
                drawContours(contourMask, std::vector<std::vector<cv::Point>>{contour}, -1, Scalar(255), FILLED);
                Moments m = moments(contour);
                cv::Point centroid(-1, -1);
                if (m.m00 != 0) centroid = Point((int)(m.m10 / m.m00), (int)(m.m01 / m.m00));
                double dist = (lastHandPos.x >= 0 && lastHandPos.y >= 0) ? norm(centroid - lastHandPos) : 0.0;
                double score = area - DIST_WEIGHT * dist;
                if (score > bestScore) {
                    bestScore = score;
                    trackedContour = contour;
                    bestCentroid = centroid;
                }
                Rect bbox = boundingRect(contour);
                //rectangle(fgMaskColor, bbox, Scalar(0, 255, 0), 2);
                //putText(fgMaskColor, "Area: " + std::to_string((int)area), bbox.tl() + Point(0, -5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            //}
        }
        if (!trackedContour.empty() && bestCentroid.x >= 0 && bestCentroid.y >= 0) {
            trajectory.push_back(bestCentroid);
            lastHandPos = bestCentroid;
            if (trajectory.size() > MAX_FRAMES) {
                trajectory.erase(trajectory.begin());
            }
        }


        double maxArea = 0;
        std::vector<cv::Point> largestContour;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contour;
            }
        }
        if (!largestContour.empty() && maxArea > MIN_AREA && maxArea < MAX_AREA) {
            Moments m = moments(largestContour);
            if (m.m00 != 0) {
                Point center((int)(m.m10 / m.m00), (int)(m.m01 / m.m00));
                trajectory.push_back(center);
                if (trajectory.size() > MAX_FRAMES) {
                    trajectory.erase(trajectory.begin());
                }
            }
        }
        // Draw the smoothed trajectory
        pathCanvas = Scalar(0,0,0);
        std::vector<cv::Point> smoothedTrajectory = smoothTrajectory(trajectory, 10);
        for (size_t i = 1; i < smoothedTrajectory.size(); ++i) {
            line(pathCanvas, smoothedTrajectory[i-1], smoothedTrajectory[i], Scalar(0,255,0), 2);
        }
        if (!smoothedTrajectory.empty()) {
            circle(pathCanvas, smoothedTrajectory.back(), 8, Scalar(0,0,255), -1);
        }
        // Mirror the image before display
        Mat mirroredMotionCap;
        cv::flip(pathCanvas, mirroredMotionCap, 1);
		Mat mirroredFrame;
		cv::flip(fgMask, mirroredFrame, 1);
        imshow("Motion Path", mirroredMotionCap);
        int key = waitKey(30);
        if (key == 'q' || key == 27) break;
        else if (key == 'c') {
            pathCanvas = Scalar(0,0,0);
            trajectory.clear();
            std::cout << "Path cleared." << std::endl;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Motion path drawing stopped." << std::endl;
    return 0;
}
