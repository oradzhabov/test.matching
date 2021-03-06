////////////////////////////////////////////////////////////////////
// File includes:
#include "ARDrawingContext.hpp"
#include "ARPipeline.hpp"
#include "DebugHelpers.hpp"
#include <PortraitObsBuilder.hpp>

////////////////////////////////////////////////////////////////////
// Standard includes:
#include <opencv2/opencv.hpp>
#include <gl/gl.h>
#include <gl/glu.h>

//
#include <filesystem>

namespace fs = std::experimental::filesystem;

/**
 * Processes a recorded video or live view from web-camera and allows you to adjust homography refinement and 
 * reprojection threshold in runtime.
 */
void processVideo(const std::vector<cv::Mat>& patternImages, CameraCalibration& calibration, cv::VideoCapture& capture);

/**
 * Processes single image. The processing goes in a loop.
 * It allows you to control the detection process by adjusting homography refinement switch and 
 * reprojection threshold in runtime.
 */
void processSingleImage(const std::vector<cv::Mat>& patternImages, CameraCalibration& calibration, const cv::Mat& image);

/**
 * Performs full detection routine on camera frame and draws the scene using drawing context.
 * In addition, this function draw overlay with debug information on top of the AR window.
 * Returns true if processing loop should be stopped; otherwise - false.
 */
bool processFrame(const cv::Mat& cameraFrame, ARPipeline& pipeline, ARDrawingContext& drawingCtx);

int main(int argc, const char * argv[])
{
    // Change this calibration to yours:
    // Note: Simply way to found calib params - http://w3.impa.br/~zang/qtcalib/nochess.html
    CameraCalibration calibration(726, 726, 320.0f, 240.0f); // My Home WebCam
    
    if (argc < 2)
    {
        std::cout << "Input image not specified" << std::endl;
        std::cout << "Usage: markerless_ar_demo <pattern list file> [filepath to recorded video or image]" << std::endl;
        return 1;
    }

    // Try to read the pattern(s):
    std::vector<cv::Mat>    patternImages;
    const fs::path          patternList(argv[1]);
    const std::wstring      ext = patternList.extension().c_str();
    if (ext != L".txt") {
        std::cout << "Input param is not a patternlist" << std::endl;
        return 2;
    }
    //
    // Read files line by line
    std::ifstream patternListFile(argv[1]);
    std::string patternImageName;
    while (std::getline(patternListFile, patternImageName)) {
            
        fs::path fullPath = patternList.parent_path();
        fullPath += fs::path("\\");
        fullPath += fs::path(patternImageName);
        //
        const std::wstring ws = fullPath.c_str();
        const std::string s(ws.begin(), ws.end());
        //
        cv::Mat img = cv::imread(s);
        if (!img.empty()) {
            /*
            cv::Mat tmp;
            double sc = 0.5;
            cv::resize(img, tmp, cv::Size((double)img.cols * sc, (double)img.rows * sc));
            img = tmp;
            */

            patternImages.push_back(img);
        }
    }

    if (argc == 2)
    {
        cv::VideoCapture cap(0);
        processVideo(patternImages, calibration, cap);
    }
    else if (argc == 3)
    {
        std::string input = argv[2];
        cv::Mat testImage = cv::imread(input);
        if (!testImage.empty())
        {
            processSingleImage(patternImages, calibration, testImage);
        }
        else 
        {
            cv::VideoCapture cap;
            if (cap.open(input))
            {
                processVideo(patternImages, calibration, cap);
            }
        }
    }
    else
    {
        std::cerr << "Invalid number of arguments passed" << std::endl;
        return 1;
    }

    return 0;
}

void processVideo(const std::vector<cv::Mat>& patternImages, CameraCalibration& calibration, cv::VideoCapture& capture)
{
    // Grab first frame to get the frame dimensions
    cv::Mat currentFrame;  
    capture >> currentFrame;

    // Check the capture succeeded:
    if (currentFrame.empty())
    {
        std::cout << "Cannot open video capture device" << std::endl;
        return;
    }

    cv::Size frameSize(currentFrame.cols, currentFrame.rows);

    ARPipeline pipeline(patternImages, calibration);
    ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);

    bool shouldQuit = false;
    do
    {
        capture >> currentFrame;
        if (currentFrame.empty())
        {
            shouldQuit = true;
            continue;
        }

        shouldQuit = processFrame(currentFrame, pipeline, drawingCtx);
    } while (!shouldQuit);
}

void processSingleImage(const std::vector<cv::Mat>& patternImages, CameraCalibration& calibration, const cv::Mat& image)
{
    cv::Size frameSize(image.cols, image.rows);
    ARPipeline pipeline(patternImages, calibration);
    ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);

    bool shouldQuit = false;
    do
    {
        shouldQuit = processFrame(image, pipeline, drawingCtx);
    } while (!shouldQuit);
}

bool processFrame(const cv::Mat& cameraFrame, ARPipeline& pipeline, ARDrawingContext& drawingCtx)
{
    // Clone image used for background (we will draw overlay on it)
    cv::Mat img = cameraFrame.clone();

    // Draw information:
    if (pipeline.m_patternDetector->enableHomographyRefinement)
        cv::putText(img, "Pose refinement: On   ('h' to switch off)", cv::Point(10,15), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));
    else
        cv::putText(img, "Pose refinement: Off  ('h' to switch on)",  cv::Point(10,15), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));

    cv::putText(img, "RANSAC threshold: " + ToString(pipeline.m_patternDetector->homographyReprojectionThreshold) + "( Use'-'/'+' to adjust)", cv::Point(10, 30), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(0,200,0));

    // Set a new camera frame:
    drawingCtx.updateBackground(img);

    // Find patterns:
    pipeline.processFrame(cameraFrame);

    //PortraitObsBuilder::Test(PortraitObsBuilder::create(pipeline));

    // Update a patterns poses found during frame processing:
    drawingCtx.patternPoses.clear();
    const size_t patternCount = pipeline.getPatternsCount();
    for (size_t i = 0; i < patternCount; i++) {
        if (pipeline.isPatternFound(i))
            drawingCtx.patternPoses.push_back(pipeline.getPatternInfo(i).pose3d);
    }

    // Request redraw of the window:
    drawingCtx.updateWindow();

    // Read the keyboard input:
    int keyCode = cv::waitKey(5); 

    bool shouldQuit = false;
    if (keyCode == '+' || keyCode == '=')
    {
        pipeline.m_patternDetector->homographyReprojectionThreshold += 0.2f;
        pipeline.m_patternDetector->homographyReprojectionThreshold = std::min(10.0f, pipeline.m_patternDetector->homographyReprojectionThreshold);
    }
    else if (keyCode == '-')
    {
        pipeline.m_patternDetector->homographyReprojectionThreshold -= 0.2f;
        pipeline.m_patternDetector->homographyReprojectionThreshold = std::max(0.0f, pipeline.m_patternDetector->homographyReprojectionThreshold);
    }
    else if (keyCode == 'h')
    {
        pipeline.m_patternDetector->enableHomographyRefinement = !pipeline.m_patternDetector->enableHomographyRefinement;
    }
    else if (keyCode == 27 || keyCode == 'q')
    {
        shouldQuit = true;
    }

    return shouldQuit;
}


