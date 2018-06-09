#include "main.h"

#include <Provider/Provider.h>
#include <State/State.h>
#include <KPDetector/KPDetector.h>


State g_state("Result");


/**
* @function on_trackbar
* @brief Callback for trackbar
*/
void on_trackbar(int, void*) {
}

int main() {
    const std::string	testCasePath = "E:/PROJECTS/CPP/OpenCV/test.matching/test-cases/tabaco/";
    //
    //const std::string   imgScenePath = "LQ\\05be0d3d5a0594cbff4f1e7ff60509f1.jpg";
    //const std::string   imgScenePath = "LQ\\shy\\1f390587fded.jpg";
    //const std::string   imgScenePath = "LQ\\shy\\238002406_1_644x461_polka-dlya-sigaret-shymkent.jpg";
    //const std::string   imgScenePath = "LQ\\angle\\18938de27b93e6be2c7150bfe4079c0a.jpg";
    //const std::string   imgScenePath = "LQ\\angle\\433995564_w640_h640_sigor.jpeg";
    //const std::string   imgScenePath = "LQ\\angle\\744858602.jpg";
    //const std::string   imgScenePath = "LQ\\siqareti.jpg";
    //const std::string   imgScenePath = "HQ\\Angle\\NYML105-828_2017_153838_hd.jpg";
    //
    //const std::string   videoScenePath = "AWM Smart Shelf on Tobacco Gondola.mp4";
    //const std::string   videoScenePath = "1.mp4";
    //const std::string   videoScenePath = "2.mp4";
    //const std::string   videoScenePath = "3.mp4";
    const std::string   videoScenePath = "4.mp4";


    //cv::Ptr<Provider>   srcProvider = Provider::CreateImageProvider(testCasePath + imgScenePath);
    cv::Ptr<Provider>   srcProvider = Provider::CreateVideoProvider(testCasePath + videoScenePath);
    if (g_state.Initialize(srcProvider) != 0)
        return -1;

    const int alpha_slider_max = 100;
    int alpha_slider = 50;
    cv::createTrackbar("TrackbarName", g_state.wndName(), &alpha_slider, alpha_slider_max, on_trackbar);

    /// Show some stuff
    on_trackbar(alpha_slider, 0);


    std::cout << "Press Esc for exit" << std::endl;
    std::cout << "Press Space for play per frame" << std::endl;
    cv::Mat         frame;
    cv::Mat         adjMtx;
    int             key = 0;
    //
    KPDetector      detectors[2];
    int             currDetector = 0;
    //
    while (key != 27) {
        if (key == ' ')
            key = cv::waitKey(0);
        else
            key = cv::waitKey(1);


        g_state.getSceneFrame(frame);
        if (frame.empty()) continue;

        detectors[currDetector].detect(frame);
        if (detectors[(currDetector + 1) % 2].haveDetection()) {
            int prevDetector = (currDetector + 1) % 2;
            int err = detectors[prevDetector].match(detectors[currDetector]);
            if (err != 0)
                currDetector = (currDetector + 1) % 2;
        }
        /*
        std::vector<cv::Vec2f>	shelfLinesArray = shelfLines(frame, CV_PI / 180.0);

        //
        bool gotIt = false;
        if (shelfLinesArray.size() > 1) {

            cv::Mat frameAndLines = frame.clone();
            for (int i = 0; i < shelfLinesArray.size(); i++)
                draw::Lines(shelfLinesArray[i], frameAndLines, CV_RGB(0, 255, 255), 1);

            //cv::Mat					img_scene_adj = AdjustPerspective(frame, shelfLinesArray, adjMtx);
            //if (img_scene_adj.empty()) continue;

            g_state.showImage(frameAndLines);
            gotIt = true;
        }

        if (!gotIt)
            g_state.showImage(frame);
        */
        currDetector = (currDetector + 1) % 2;
    }

    return 0;
}