#include <main.h>

cv::Mat AdjustPerspective(const cv::Mat & img_scene, std::vector<cv::Vec2f> & lines, cv::Mat & perspectiveTranformMatrix) {

    if (lines.size() > 1) {

        // Determine top/bottom corners of shelf lines
        const cv::Vec2f & topLine = lines.front();
        const cv::Vec2f & bottomLine = lines.back();
        cv::Point2f ptTopLeft = topLine[1] != 0 ? cv::Point2f(0, topLine[0] / sin(topLine[1])) :
            cv::Point2f(topLine[0], 0);
        cv::Point2f ptTopRight = topLine[1] != 0 ? cv::Point2f(img_scene.size().width, -1 / tan(topLine[1])*img_scene.size().width + topLine[0] / sin(topLine[1])) :
            cv::Point2f(topLine[0], img_scene.size().height);
        //
        cv::Point2f ptBottomLeft = bottomLine[1] != 0 ? cv::Point2f(0, bottomLine[0] / sin(bottomLine[1])) :
            cv::Point2f(bottomLine[0], 0);
        cv::Point2f ptBottomRight = bottomLine[1] != 0 ? cv::Point2f(img_scene.size().width, -1 / tan(bottomLine[1])*img_scene.size().width + bottomLine[0] / sin(bottomLine[1])) :
            cv::Point2f(bottomLine[0], img_scene.size().height);
        //
        cv::Vec2f topEdge = ptTopRight - ptTopLeft;
        cv::Vec2f bottomEdge = ptBottomRight - ptBottomLeft;
        cv::Vec2f leftEdge = ptBottomLeft - ptTopLeft;
        cv::Vec2f rightEdge = ptBottomRight - ptTopRight;
        /*
        * Tune perspective distortion of shelf length
        *
        * b - distance(in pixels) from camera to 'a'
        * a/b = tan(alpha);
        * c/b = tan(betta) = a / (b + d);
        * hence: (b + d) = a*b/c => d = a*b/c - b => d = b*(a-c)/c;
        * f = sqrt(g^2 + d^2); - real horizontal length
        */
        float g = img_scene.cols;
        float a = cv::max<float>(cv::norm(leftEdge), cv::norm(rightEdge));
        float c = cv::min<float>(cv::norm(leftEdge), cv::norm(rightEdge));
        float b = img_scene.cols * 2; // good average default value for vertical(portrait) camera position
        float f = sqrt(g*g + pow(b*(a - c) / c, 2));
        float maxLength = f;
        float maxHeight = a;
        //
        //
        //
        std::vector<cv::Point2f>	src(4), dst(4);
        src[0] = ptTopLeft;			dst[0] = cv::Point2f(0, 0);
        src[1] = ptTopRight;        dst[1] = cv::Point2f(maxLength - 1, 0);
        src[2] = ptBottomRight;     dst[2] = cv::Point2f(maxLength - 1, maxHeight - 1);
        src[3] = ptBottomLeft;      dst[3] = cv::Point2f(0, maxHeight - 1);
        //
        cv::Mat undistorted = cv::Mat(maxHeight, maxLength, CV_8UC1);
        //
        //cv::Mat perspectiveTranformMatrix = cv::getPerspectiveTransform(src, dst);
        //cv::Mat perspectiveTranformMatrix = cv::findHomography(src, dst);
        perspectiveTranformMatrix = cv::findHomography(src, dst);

        /*
        double p[] = { 0, 0, 1,
        img_scene.cols - 1, 0, 1,
        img_scene.cols - 1, img_scene.rows - 1, 1,
        0, img_scene.rows - 1, 1 };
        cv::Mat t(4,3,CV_64FC1, &p);
        cv::Mat t2 = perspectiveTranformMatrix * t.t();
        t2 = t2.t();
        std::cout << t2 << std::endl;
        */



        cv::warpPerspective(img_scene, undistorted, perspectiveTranformMatrix, undistorted.size());
        //cv::imshow("undistorted", undistorted);
        //
        /*
        cv::Mat undistortedInv;
        cv::warpPerspective(undistorted, undistortedInv, perspectiveTranformMatrix.inv(), img_scene.size());
        cv::imshow("undistortedInv", undistortedInv);
        cv::waitKey(0);
        */

        return undistorted;
    }
    return cv::Mat();
}