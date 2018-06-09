#include <main.h>

namespace draw {

    void Lines(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb, const int thickness) {

        if (line[1] != 0) {
            float m = -1 / tan(line[1]);
            float c = line[0] / sin(line[1]);

            cv::line(img, cv::Point(0, c), cv::Point(img.size().width, m*img.size().width + c), rgb, thickness);
        } else {
            cv::line(img, cv::Point(line[0], 0), cv::Point(line[0], img.size().height), rgb, thickness);
        }
    }
}