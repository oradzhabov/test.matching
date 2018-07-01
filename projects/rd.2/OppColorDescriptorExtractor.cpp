#include "OppColorDescriptorExtractor.h"

using namespace cv;
using namespace std;

const cv::String OppColorDescriptorExtractor::DefaultName = cv::String("OppColorDescriptorExtractor");

/****************************************************************************************\
*                             OppColorDescriptorExtractor                           *
\****************************************************************************************/
OppColorDescriptorExtractor::OppColorDescriptorExtractor(const Ptr<DescriptorExtractor>& _descriptorExtractor) :
	descriptorExtractor(_descriptorExtractor)
{
	CV_Assert(!descriptorExtractor.empty());
}

static void convertBGRImageToOpponentColorSpace(const Mat& bgrImage, vector<Mat>& opponentChannels) {
	if (bgrImage.type() != CV_8UC3)
		CV_Error(CV_StsBadArg, "input image must be an BGR image of type CV_8UC3");

    {
        //cv::split(bgrImage, opponentChannels);
        
        cv::Mat diffSpaceImg;
        cv::cvtColor(bgrImage, diffSpaceImg, CV_BGR2HSV); // For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
        //cv::cvtColor(bgrImage, diffSpaceImg, CV_BGR2Luv);
        cv::split(diffSpaceImg, opponentChannels);
        opponentChannels[0] *= 255.0 / 179.0;

        opponentChannels[0] = 0;
        //opponentChannels[1] = 0;
        //opponentChannels[2] = 0;
        if (false)
        {
            double min, max;
            cv::minMaxLoc(opponentChannels[0], &min, &max);
            cv::minMaxLoc(opponentChannels[1], &min, &max);
            cv::minMaxLoc(opponentChannels[2], &min, &max);
        }

        return;
    }

	// Prepare opponent color space storage matrices.
	opponentChannels.resize(3);
	opponentChannels[0] = cv::Mat(bgrImage.size(), CV_8UC1); // R-G RED-GREEN
	opponentChannels[1] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G-2B YELLOW-BLUE
	opponentChannels[2] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G+B

	for (int y = 0; y < bgrImage.rows; ++y)
		for (int x = 0; x < bgrImage.cols; ++x)
		{
			Vec3b v = bgrImage.at<Vec3b>(y, x);
			uchar& b = v[0];
			uchar& g = v[1];
			uchar& r = v[2];

			opponentChannels[0].at<uchar>(y, x) = saturate_cast<uchar>(0.5f    * (255 + g - r));       // (R - G)/sqrt(2), but converted to the destination data type
			opponentChannels[1].at<uchar>(y, x) = saturate_cast<uchar>(0.25f   * (510 + r + g - 2 * b)); // (R + G - 2B)/sqrt(6), but converted to the destination data type
			opponentChannels[2].at<uchar>(y, x) = saturate_cast<uchar>(1.f / 3.f * (r + g + b));         // (R + G + B)/sqrt(3), but converted to the destination data type
		}
}

struct KP_LessThan
{
	KP_LessThan(const vector<KeyPoint>& _kp) : kp(&_kp) {}
	bool operator()(int i, int j) const
	{
		return (*kp)[i].class_id < (*kp)[j].class_id;
	}
	const vector<KeyPoint>* kp;
};

void OppColorDescriptorExtractor::computeImpl(const Mat& bgrImage, vector<KeyPoint>& keypoints, Mat& descriptors) const
{
	vector<Mat> opponentChannels;
	convertBGRImageToOpponentColorSpace(bgrImage, opponentChannels);

	const int N = 3; // channels count
	vector<KeyPoint> channelKeypoints[N];
	Mat channelDescriptors[N];
	vector<int> idxs[N];

	// Compute descriptors three times, once for each Opponent channel to concatenate into a single color descriptor
	int maxKeypointsCount = 0;
	for (int ci = 0; ci < N; ci++)
	{
		channelKeypoints[ci].insert(channelKeypoints[ci].begin(), keypoints.begin(), keypoints.end());
		// Use class_id member to get indices into initial keypoints vector
		for (size_t ki = 0; ki < channelKeypoints[ci].size(); ki++)
			channelKeypoints[ci][ki].class_id = (int)ki;

		descriptorExtractor->compute(opponentChannels[ci], channelKeypoints[ci], channelDescriptors[ci]);
		idxs[ci].resize(channelKeypoints[ci].size());
		for (size_t ki = 0; ki < channelKeypoints[ci].size(); ki++)
		{
			idxs[ci][ki] = (int)ki;
		}
		std::sort(idxs[ci].begin(), idxs[ci].end(), KP_LessThan(channelKeypoints[ci]));
		maxKeypointsCount = std::max(maxKeypointsCount, (int)channelKeypoints[ci].size());
	}

	vector<KeyPoint> outKeypoints;
	outKeypoints.reserve(keypoints.size());

	int dSize = descriptorExtractor->descriptorSize();
	Mat mergedDescriptors(maxKeypointsCount, 3 * dSize, descriptorExtractor->descriptorType());
	int mergedCount = 0;
	// cp - current channel position
	size_t cp[] = { 0, 0, 0 };
	while (cp[0] < channelKeypoints[0].size() &&
		cp[1] < channelKeypoints[1].size() &&
		cp[2] < channelKeypoints[2].size())
	{
		const int maxInitIdx = std::max(0, std::max(channelKeypoints[0][idxs[0][cp[0]]].class_id,
			std::max(channelKeypoints[1][idxs[1][cp[1]]].class_id,
				channelKeypoints[2][idxs[2][cp[2]]].class_id)));

		while (channelKeypoints[0][idxs[0][cp[0]]].class_id < maxInitIdx && cp[0] < channelKeypoints[0].size()) { cp[0]++; }
		while (channelKeypoints[1][idxs[1][cp[1]]].class_id < maxInitIdx && cp[1] < channelKeypoints[1].size()) { cp[1]++; }
		while (channelKeypoints[2][idxs[2][cp[2]]].class_id < maxInitIdx && cp[2] < channelKeypoints[2].size()) { cp[2]++; }
		if (cp[0] >= channelKeypoints[0].size() || cp[1] >= channelKeypoints[1].size() || cp[2] >= channelKeypoints[2].size())
			break;

		if (channelKeypoints[0][idxs[0][cp[0]]].class_id == maxInitIdx &&
			channelKeypoints[1][idxs[1][cp[1]]].class_id == maxInitIdx &&
			channelKeypoints[2][idxs[2][cp[2]]].class_id == maxInitIdx)
		{
			outKeypoints.push_back(keypoints[maxInitIdx]);
			// merge descriptors
			for (int ci = 0; ci < N; ci++)
			{
				Mat dst = mergedDescriptors(Range(mergedCount, mergedCount + 1), Range(ci*dSize, (ci + 1)*dSize));
				channelDescriptors[ci].row(idxs[ci][cp[ci]]).copyTo(dst);
				cp[ci]++;
			}
			mergedCount++;
		}
	}
	mergedDescriptors.rowRange(0, mergedCount).copyTo(descriptors);
	std::swap(outKeypoints, keypoints);
}

void OppColorDescriptorExtractor::detectAndCompute(InputArray bgrImage, InputArray ,
	std::vector<KeyPoint>& keypoints,
	OutputArray descriptors,
	bool e)
{
	computeImpl(bgrImage.getMat(), keypoints, descriptors.getMatRef());
}

void OppColorDescriptorExtractor::read(const FileNode& fn)
{
	descriptorExtractor->read(fn);
}

void OppColorDescriptorExtractor::write(FileStorage& fs) const
{
	descriptorExtractor->write(fs);
}

int OppColorDescriptorExtractor::descriptorSize() const
{
	return 3 * descriptorExtractor->descriptorSize();
}

int OppColorDescriptorExtractor::descriptorType() const
{
	return descriptorExtractor->descriptorType();
}

bool OppColorDescriptorExtractor::empty() const
{
	return descriptorExtractor.empty() || (DescriptorExtractor*)(descriptorExtractor)->empty();
}