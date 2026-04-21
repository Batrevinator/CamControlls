#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv; 


void SharpenImageManual(Mat& inImage, Mat& outImage){
    CV_Assert(inImage.depth() == CV_8U);

    int nChannels = inImage.channels();
    outImage.create(inImage.size(), inImage.type());

    for(int i = 1; i <inImage.rows-1; ++i){
        const uchar* prev = inImage.ptr<uchar>(i-1);
	const uchar* cur = inImage.ptr<uchar>(i);
	const uchar* next = inImage.ptr<uchar>(i+1);

	uchar* output = outImage.ptr<uchar>(i);

	for(int j = nChannels; j < nChannels*(inImage.cols-1); ++j){
	    output[j] = saturate_cast<uchar>(5*cur[j] - cur[j-nChannels] - cur[j+nChannels] - prev[j] -next[j]);
	}
    } 

    outImage.row(0).setTo(Scalar(0));
    outImage.row(outImage.rows-1).setTo(Scalar(0));
    outImage.col(0).setTo(Scalar(0));
    outImage.col(outImage.cols-1).setTo(Scalar(0));
}

Mat& ScanImageAndReduceC(Mat& image, const uchar* const table){

    //A check for the type of matrix (in this case char)	
    CV_Assert(image.depth() == CV_8U);

    int channels = image.channels();
    int numRows = image.rows;
    int numCols = image.cols * channels;

    if(image.isContinuous()){
    	numCols *= numRows;
	numRows = 1;
    }

    int iteratorX, iteratorY;

    uchar* p;
    for (iteratorX = 0; iteratorX < numRows; ++iteratorX){
        p = image.ptr<uchar>(iteratorX);
	for(iteratorY = 0; iteratorY < numCols; ++iteratorY){
	    p[iteratorY] = table[p[iteratorY]];
	}
    }
    return image;

}

int main(int argc, char** argv)
{
    cout << "Motion Capture System" << endl;
    cout << "=====================" << endl;
    
    // Try different camera indices
    VideoCapture cap;
    bool cameraFound = false;
    int cameraIndex = -1;
    
    // Try camera indices from 0 to 9
    for (int i = 0; i < 10; ++i) {
        cap.open(i);
        if (cap.isOpened()) {
            cameraIndex = i;
            cameraFound = true;
            cout << "Camera found at index " << i << endl;
            break;
        }
        cap.release();
    }
    
    if (!cameraFound) {
        cerr << "Error: No camera found!" << endl;
        cerr << "Possible solutions:" << endl;
        cerr << "1. Check if you're running in WSL - cameras don't work directly in WSL" << endl;
        cerr << "2. If in WSL, run this application from Windows Command Prompt instead" << endl;
        cerr << "3. Make sure your camera is connected and not used by another application" << endl;
        cerr << "4. Try running: sudo chmod 666 /dev/video*" << endl;
        cerr << "5. Check camera permissions with: ls -la /dev/video*" << endl;
        return -1;
    }
    
    cout << "Using camera index: " << cameraIndex << endl;
    
    // Get camera properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    
    cout << "Camera properties:" << endl;
    cout << "  Resolution: " << width << "x" << height << endl;
    cout << "  FPS: " << fps << endl;
    
    // Create background subtractor
    Ptr<BackgroundSubtractorMOG2> bgSubtractor = createBackgroundSubtractorMOG2();
    
    Mat frame, fgMask, blurred;
    
    // Create windows
    namedWindow("Live Feed", WINDOW_AUTOSIZE);
    namedWindow("Motion Detection", WINDOW_AUTOSIZE);
    
    cout << endl << "Controls:" << endl;
    cout << "  'q' or ESC: Quit" << endl;
    cout << "  's': Save current frame" << endl;
    cout << "  'c': Clear background model" << endl;
    cout << endl << "Starting motion capture..." << endl;
    
    while (true) {
        // Capture frame
        cap >> frame;
        
        if (frame.empty()) {
            cerr << "Error: Could not capture frame" << endl;
            break;
        }
        
        // Apply background subtraction
        bgSubtractor->apply(frame, fgMask);
        
        // Apply morphological operations to reduce noise
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
        morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel);
        
        // Apply Gaussian blur to smooth the mask
        GaussianBlur(fgMask, blurred, Size(11, 11), 0);
        
        // Threshold to get binary motion mask
        threshold(blurred, fgMask, 50, 255, THRESH_BINARY);
        
        // Display frames
        imshow("Live Feed", frame);
        imshow("Motion Detection", fgMask);
        
        // Check for key press
        int key = waitKey(30);
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 's') {
            string filename = "motion_capture_" + to_string(time(0)) + ".png";
            imwrite(filename, frame);
            cout << "Frame saved as: " << filename << endl;
        } else if (key == 'c') {
            // Clear background model
            bgSubtractor = createBackgroundSubtractorMOG2();
            cout << "Background model cleared" << endl;
        }
    }
    
    cap.release();
    destroyAllWindows();
    
    cout << "Motion capture stopped." << endl;
    return 0;
}
