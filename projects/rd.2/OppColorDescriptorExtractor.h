#pragma once

#include <opencv2/opencv.hpp>

class OppColorDescriptorExtractor : public cv::DescriptorExtractor
{
public:
    static const cv::String DefaultName;

	OppColorDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& _descriptorExtractor);

protected:
	void computeImpl(const cv::Mat& bgrImage, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

	void read(const cv::FileNode& fn);
	void write(cv::FileStorage& fs) const;
	int OppColorDescriptorExtractor::descriptorSize() const;
	int OppColorDescriptorExtractor::descriptorType() const;
	bool OppColorDescriptorExtractor::empty() const;

	virtual cv::String getDefaultName() const { return OppColorDescriptorExtractor::DefaultName; }

	void detectAndCompute(cv::InputArray a, cv::InputArray b,
		std::vector<cv::KeyPoint>& c,
		cv::OutputArray d,
		bool e);
private:
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
};

