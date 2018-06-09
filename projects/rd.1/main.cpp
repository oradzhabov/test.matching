#include "main.h"

#include <Provider/Provider.h>
#include <State/State.h>


State g_state("Result");


/**
* @function on_trackbar
* @brief Callback for trackbar
*/
void on_trackbar(int, void*)
{
    cv::Mat frame;
    g_state.getSceneFrame(frame);

    cv::Mat adjMtx;
    std::vector<cv::Vec2f>	shelfLinesArray = shelfLines(frame);
    cv::Mat					img_scene_adj = AdjustPerspective(frame, shelfLinesArray, adjMtx);

    g_state.showImage(img_scene_adj);
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
    const std::string   imgScenePath = "HQ\\Angle\\NYML105-828_2017_153838_hd.jpg";


    if (g_state.Initialize(Provider::CreateImageProvider(testCasePath + imgScenePath)) != 0)
        return -1;

    const int alpha_slider_max = 100;
    int alpha_slider = 50;
    cv::createTrackbar("TrackbarName", g_state.wndName(), &alpha_slider, alpha_slider_max, on_trackbar);

    /// Show some stuff
    on_trackbar(alpha_slider, 0);

    std::cout << "Press Esc for exit" << std::endl;
    while (cv::waitKey(0) != 27) {}

    return 0;
}