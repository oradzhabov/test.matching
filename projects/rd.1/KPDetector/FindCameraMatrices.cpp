#include <main.h>

using namespace cv;
using namespace std;

#ifdef USE_EIGEN
#include <Eigen/Eigen>
#endif

#define DECOMPOSE_SVD

#ifndef CV_PCA_DATA_AS_ROW
#define CV_PCA_DATA_AS_ROW 0
#endif




std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
    std::vector<cv::Point3d> out;
    for (unsigned int i = 0; i<cpts.size(); i++)
        out.push_back(cpts[i].pt);

    return out;
}

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
    ps.clear();
    for (unsigned int i = 0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void PointsToKeyPoints(const vector<Point2f>& ps, vector<KeyPoint>& kps) {
    kps.clear();
    for (unsigned int i = 0; i<ps.size(); i++) kps.push_back(KeyPoint(ps[i], 1.0f));
}

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
    Matx34d P,		//camera 1 matrix
    Point3d u1,		//homogenous image point in 2nd camera
    Matx34d P1		//camera 2 matrix
)
{

    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    //	cout << "u " << u <<", u1 " << u1 << endl;
    //	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u.x*P(1)-u.y*P(0);
    //	A(3) = u1.x*P1(2)-P1(0);
    //	A(4) = u1.y*P1(2)-P1(1);
    //	A(5) = u1.x*P(1)-u1.y*P1(0);
    //	Matx43d A; //not working for some reason...
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u1.x*P1(2)-P1(0);
    //	A(3) = u1.y*P1(2)-P1(1);
    Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
        u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
        u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
        u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
    );
    Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
        -(u.y*P(2, 3) - P(1, 3)),
        -(u1.x*P1(2, 3) - P1(0, 3)),
        -(u1.y*P1(2, 3) - P1(1, 3)));

    Mat_<double> X;
    solve(A, B, X, DECOMP_SVD);

    return X;
}

#define EPSILON 0.0001

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
    Matx34d P,			//camera 1 matrix
    Point3d u1,			//homogenous image point in 2nd camera
    Matx34d P1			//camera 2 matrix
) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4, 1);
    for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
            (u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
            (u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
            (u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
        );
        Mat_<double> B = (Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
            -(u.y*P(2, 3) - P(1, 3)) / wi,
            -(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
            -(u1.y*P1(2, 3) - P1(1, 3)) / wi1
            );

        solve(A, B, X_, DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }
    return X;
}

//Triagulate points
double TriangulatePoints(const vector<KeyPoint>& pt_set1,
    const vector<KeyPoint>& pt_set2,
    const Mat& K,
    const Mat& Kinv,
    const Mat& distcoeff,
    const Matx34d& P,
    const Matx34d& P1,
    vector<CloudPoint>& pointcloud,
    vector<KeyPoint>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
    vector<double> depths;
#endif

    //	pointcloud.clear();
    correspImg1Pt.clear();

    Matx44d P1_(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
        P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
        P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
        0, 0, 0, 1);
    Matx44d P1inv(P1_.inv());

    cout << "Triangulating...";
    double t = getTickCount();
    vector<double> reproj_error;
    unsigned int pts_size = pt_set1.size();

#if 0
    //Using OpenCV's triangulation
    //convert to Point2f
    vector<Point2f> _pt_set1_pt, _pt_set2_pt;
    KeyPointsToPoints(pt_set1, _pt_set1_pt);
    KeyPointsToPoints(pt_set2, _pt_set2_pt);

    //undistort
    Mat pt_set1_pt, pt_set2_pt;
    undistortPoints(_pt_set1_pt, pt_set1_pt, K, distcoeff);
    undistortPoints(_pt_set2_pt, pt_set2_pt, K, distcoeff);

    //triangulate
    Mat pt_set1_pt_2r = pt_set1_pt.reshape(1, 2);
    Mat pt_set2_pt_2r = pt_set2_pt.reshape(1, 2);
    Mat pt_3d_h(1, pts_size, CV_32FC4);
    cv::triangulatePoints(P, P1, pt_set1_pt_2r, pt_set2_pt_2r, pt_3d_h);

    //calculate reprojection
    vector<Point3f> pt_3d;
    convertPointsHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
    cv::Mat_<double> R = (cv::Mat_<double>(3, 3) << P(0, 0), P(0, 1), P(0, 2), P(1, 0), P(1, 1), P(1, 2), P(2, 0), P(2, 1), P(2, 2));
    Vec3d rvec; Rodrigues(R, rvec);
    Vec3d tvec(P(0, 3), P(1, 3), P(2, 3));
    vector<Point2f> reprojected_pt_set1;
    projectPoints(pt_3d, rvec, tvec, K, distcoeff, reprojected_pt_set1);

    for (unsigned int i = 0; i<pts_size; i++) {
        CloudPoint cp;
        cp.pt = pt_3d[i];
        pointcloud.push_back(cp);
        reproj_error.push_back(norm(_pt_set1_pt[i] - reprojected_pt_set1[i]));
    }
#else
    Mat_<double> KP1 = K * Mat(P1);
#pragma omp parallel for num_threads(1)
    for (int i = 0; i<pts_size; i++) {
        Point2f kp = pt_set1[i].pt;
        Point3d u(kp.x, kp.y, 1.0);
        Mat_<double> um = Kinv * Mat_<double>(u);
        u.x = um(0); u.y = um(1); u.z = um(2);

        Point2f kp1 = pt_set2[i].pt;
        Point3d u1(kp1.x, kp1.y, 1.0);
        Mat_<double> um1 = Kinv * Mat_<double>(u1);
        u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

        Mat_<double> X = IterativeLinearLSTriangulation(u, P, u1, P1);

        //		cout << "3D Point: " << X << endl;
        //		Mat_<double> x = Mat(P1) * X;
        //		cout <<	"P1 * Point: " << x << endl;
        //		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
        //		cout <<	"Point: " << xPt << endl;
        Mat_<double> xPt_img = KP1 * X;				//reproject
                                                    //		cout <<	"Point * K: " << xPt_img << endl;
        Point2f xPt_img_(xPt_img(0) / xPt_img(2), xPt_img(1) / xPt_img(2));

#pragma omp critical
        {
            double reprj_err = norm(xPt_img_ - kp1);
            reproj_error.push_back(reprj_err);

            CloudPoint cp;
            cp.pt = Point3d(X(0), X(1), X(2));
            cp.reprojection_error = reprj_err;

            pointcloud.push_back(cp);
            correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM__DEBUG__
            depths.push_back(X(2));
#endif
        }
    }
#endif

    Scalar mse = mean(reproj_error);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Done. (" << pointcloud.size() << "points, " << t << "sec, mean reproj err = " << mse[0] << ")" << endl;

    //show "range image"
#ifdef __SFM__DEBUG__
    {
        double minVal, maxVal;
        minMaxLoc(depths, &minVal, &maxVal);
        Mat tmp(240, 320, CV_8UC3, Scalar(0, 0, 0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
        for (unsigned int i = 0; i<pointcloud.size(); i++) {
            double _d = std::max<double>(std::min<double>((pointcloud[i].pt.z - minVal) / (maxVal - minVal), 1.0), 0.0);
            circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0 - (_d)), 255, 255), CV_FILLED);
        }
        cvtColor(tmp, tmp, CV_HSV2BGR);
        imshow("Depth Map", tmp);
        //waitKey(0);
        //destroyWindow("Depth Map");
    }
#endif

    return mse[0];
}

void DecomposeEssentialUsingHorn90(double _E[9], double _R1[9], double _R2[9], double _t1[3], double _t2[3]) {
	//from : http://people.csail.mit.edu/bkph/articles/Essential.pdf
#ifdef USE_EIGEN
	using namespace Eigen;

	Matrix3d E = Map<Matrix<double,3,3,RowMajor> >(_E);
	Matrix3d EEt = E * E.transpose();
	Vector3d e0e1 = E.col(0).cross(E.col(1)),e1e2 = E.col(1).cross(E.col(2)),e2e0 = E.col(2).cross(E.col(2));
	Vector3d b1,b2;

#if 1
	//Method 1
	Matrix3d bbt = 0.5 * EEt.trace() * Matrix3d::Identity() - EEt; //Horn90 (12)
	Vector3d bbt_diag = bbt.diagonal();
	if (bbt_diag(0) > bbt_diag(1) && bbt_diag(0) > bbt_diag(2)) {
		b1 = bbt.row(0) / sqrt(bbt_diag(0));
		b2 = -b1;
	} else if (bbt_diag(1) > bbt_diag(0) && bbt_diag(1) > bbt_diag(2)) {
		b1 = bbt.row(1) / sqrt(bbt_diag(1));
		b2 = -b1;
	} else {
		b1 = bbt.row(2) / sqrt(bbt_diag(2));
		b2 = -b1;
	}
#else
	//Method 2
	if (e0e1.norm() > e1e2.norm() && e0e1.norm() > e2e0.norm()) {
		b1 = e0e1.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else if (e1e2.norm() > e0e1.norm() && e1e2.norm() > e2e0.norm()) {
		b1 = e1e2.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else {
		b1 = e2e0.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	}
#endif
	
	//Horn90 (19)
	Matrix3d cofactors; cofactors.col(0) = e1e2; cofactors.col(1) = e2e0; cofactors.col(2) = e0e1;
	cofactors.transposeInPlace();
	
	//B = [b]_x , see Horn90 (6) and http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
	Matrix3d B1; B1 <<	0,-b1(2),b1(1),
						b1(2),0,-b1(0),
						-b1(1),b1(0),0;
	Matrix3d B2; B2 <<	0,-b2(2),b2(1),
						b2(2),0,-b2(0),
						-b2(1),b2(0),0;

	Map<Matrix<double,3,3,RowMajor> > R1(_R1),R2(_R2);

	//Horn90 (24)
	R1 = (cofactors.transpose() - B1*E) / b1.dot(b1);
	R2 = (cofactors.transpose() - B2*E) / b2.dot(b2);
	Map<Vector3d> t1(_t1),t2(_t2); 
	t1 = b1; t2 = b2;
	
	cout << "Horn90 provided " << endl << R1 << endl << "and" << endl << R2 << endl;
#endif
}

bool CheckCoherentRotation(cv::Mat_<double>& R) {

	
	if(fabsf(determinant(R))-1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}

Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
					   const vector<KeyPoint>& imgpts2,
					   vector<KeyPoint>& imgpts1_good,
					   vector<KeyPoint>& imgpts2_good
#ifdef __SFM__DEBUG__
					  ,const Mat& img_1,
					  const Mat& img_2
#endif
					  ) 
{
	//Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	vector<uchar> status(imgpts1.size());
	
	imgpts1_good.clear(); imgpts2_good.clear();
	
//    vector<KeyPoint> imgpts1_tmp = imgpts1;
//    vector<KeyPoint> imgpts2_tmp = imgpts2;
	
	Mat F;
	{
		vector<Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1, pts1);
		KeyPointsToPoints(imgpts2, pts2);
#ifdef __SFM__DEBUG__
		cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1.size() << ")" << endl;
		cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2.size() << ")" << endl;
#endif
		double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
        //F = findFundamentalMat(pts1, pts2, FM_RANSAC,10,0.99,status);
	}
	
//	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1[i]);
			imgpts2_good.push_back(imgpts2[i]);
/*
			if (matches.size() <= 0) { //points already aligned...
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			} else {
				new_matches.push_back(matches[i]);
			}
*/
		}
	}	
	
	return F;
}

void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
#if 1
	//Using OpenCV's SVD
	SVD svd(E,SVD::MODIFY_A); // ros: was SVD::MODIFY_A
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
#else
	//Using Eigen's SVD
	cout << "Eigen3 SVD..\n";
	Eigen::Matrix3f  e = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >((double*)E.data).cast<float>();
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf Esvd_u = svd.matrixU();
	Eigen::MatrixXf Esvd_v = svd.matrixV();
	svd_u = (Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
						  Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
						  Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
	Mat_<double> svd_v = (Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
						  Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
						  Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
	svd_vt = svd_v.t();
	svd_w = (Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
#endif
/*	
	cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	cout << "----------------------------------------------------\n";
    */
}

double TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) {
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye(); 
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.25) //ros: was 0.75
		return percentage; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			Vec3d w = Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return 0;
	}

	return percentage;
}

bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2) 
{
#ifdef DECOMPOSE_SVD
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio > 1.0) 
        singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]

	if (singular_values_ratio < 0.7) { // ros: was 0.7
		cout << "singular values are too far apart\n";
        cout << svd_w << endl;
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
#else
	//Using Horn E decomposition
	DecomposeEssentialUsingHorn90(E[0],R1[0],R2[0],t1[0],t2[0]);
#endif
	return true;
}

bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv, 
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1out,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
						,const Mat& img_1,
						const Mat& img_2
#endif
						) 
{
	cout << "Find camera matrices...";
	double t = getTickCount();
		
	Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good
#ifdef __SFM__DEBUG__
                            ,img_1,img_2
#endif
                            );

	if(imgpts1_good.size() < 8) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
		cerr << "not enough inliers after F matrix" << endl;
		return false;
	}
		
	// Essential matrix: compute then extract cameras [R|t]
	Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

	//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
	if(fabsf(determinant(E)) > 1e-07) {
		cout << "det(E) != 0 : " << determinant(E) << "\n";
		P1out = 0;
		return false;
	}
		
	Mat_<double> R1(3,3);
	Mat_<double> R2(3,3);
	Mat_<double> t1(1,3);
	Mat_<double> t2(1,3);

	//decompose E to P' , HZ (9.19)
	{		
		if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

		//if(determinant(R1) + 1.0 < std::numeric_limits<double>::min()) {
		//if (std::abs(determinant(R1) + 1.0) < std::numeric_limits<double>::min()) {
		//if (std::abs(determinant(R1) + 1.0) < std::numeric_limits<double>::epsilon()) {
		if (std::abs(determinant(R1) + 1.0) < 1.e-9) {
			//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
			std::cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
			E = -E;
			DecomposeEtoRandT(E,R1,R2,t1,t2);
		}
		/*
		if (!CheckCoherentRotation(R1)) {
			cout << "resulting rotation is not coherent\n";
			P1out = 0;
			return false;
		}
		*/
		/*
		P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
						*/
		//cout << "Testing P1 " << endl << Mat(P1) << endl;
			
		vector<CloudPoint>  pcloud1;
        vector<KeyPoint>    corresp;

		struct TriangleResult {
			Matx34d				P1;
			double              reproj_error1;
			double              reproj_error2;
			double				percent1;
			double				percent2;
			bool				checkCoherent;
			vector<CloudPoint>  pcloud;
		};
		vector<uchar>       tmp_status;
		TriangleResult		tstResult[4];
		tstResult[0].P1 = Matx34d(R1(0, 0), R1(0, 1), R1(0, 2), t1(0),
									R1(1, 0), R1(1, 1), R1(1, 2), t1(1),
									R1(2, 0), R1(2, 1), R1(2, 2), t1(2));
		tstResult[0].checkCoherent = CheckCoherentRotation(R1);
		tstResult[0].reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, tstResult[0].P1, tstResult[0].pcloud, corresp);
		tstResult[0].reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, tstResult[0].P1, P, pcloud1, corresp);
		tstResult[0].percent1 = TestTriangulation(tstResult[0].pcloud, tstResult[0].P1, tmp_status);
		tstResult[0].percent2 = TestTriangulation(pcloud1, P, tmp_status);
		//
		pcloud1.clear(); corresp.clear();
		tstResult[1].P1 = Matx34d(R1(0, 0), R1(0, 1), R1(0, 2), t2(0),
									R1(1, 0), R1(1, 1), R1(1, 2), t2(1),
									R1(2, 0), R1(2, 1), R1(2, 2), t2(2));
		tstResult[1].checkCoherent = tstResult[0].checkCoherent;
		tstResult[1].reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, tstResult[1].P1, tstResult[1].pcloud, corresp);
		tstResult[1].reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, tstResult[1].P1, P, pcloud1, corresp);
		tstResult[1].percent1 = TestTriangulation(tstResult[1].pcloud, tstResult[1].P1, tmp_status);
		tstResult[1].percent2 = TestTriangulation(pcloud1, P, tmp_status);
		//
		pcloud1.clear(); corresp.clear();
		tstResult[2].P1 = Matx34d(R2(0, 0), R2(0, 1), R2(0, 2), t1(0),
									R2(1, 0), R2(1, 1), R2(1, 2), t1(1),
									R2(2, 0), R2(2, 1), R2(2, 2), t1(2));
		tstResult[2].checkCoherent = CheckCoherentRotation(R2);
		tstResult[2].reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, tstResult[2].P1, tstResult[2].pcloud, corresp);
		tstResult[2].reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, tstResult[2].P1, P, pcloud1, corresp);
		tstResult[2].percent1 = TestTriangulation(tstResult[2].pcloud, tstResult[2].P1, tmp_status);
		tstResult[2].percent2 = TestTriangulation(pcloud1, P, tmp_status);
		//
		pcloud1.clear(); corresp.clear();
		tstResult[3].P1 = Matx34d(R2(0, 0), R2(0, 1), R2(0, 2), t2(0),
									R2(1, 0), R2(1, 1), R2(1, 2), t2(1),
									R2(2, 0), R2(2, 1), R2(2, 2), t2(2));
		tstResult[3].checkCoherent = tstResult[2].checkCoherent;
		tstResult[3].reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, tstResult[3].P1, tstResult[3].pcloud, corresp);
		tstResult[3].reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, tstResult[3].P1, P, pcloud1, corresp);
		tstResult[3].percent1 = TestTriangulation(tstResult[3].pcloud, tstResult[3].P1, tmp_status);
		tstResult[3].percent2 = TestTriangulation(pcloud1, P, tmp_status);

		int bestI = -1;
		double bestPercent = 0;
		double minError = std::numeric_limits<double>::max();
		for (int i = 0; i < 4; ++i) {
			if (tstResult[i].checkCoherent) {
				double minP = std::min<double>(tstResult[i].percent1, tstResult[i].percent2);
				double maxE = std::min<double>(tstResult[i].reproj_error1, tstResult[i].reproj_error2);
				if (minP >= bestPercent && maxE < minError) {
					minError = maxE;
					bestPercent = minP;
					bestI = i;
				}
			}
		}
		if (bestI < 0) {
			std::cout << "Shit." << endl;
			return false;
		}
		P1out = tstResult[bestI].P1;
		//

		/*
		double              reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
		double              reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
        //
		//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
		if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
			P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
			//cout << "Testing P1 "<< endl << Mat(P1) << endl;

			pcloud.clear(); pcloud1.clear(); corresp.clear();
			reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
			reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
				
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				if (!CheckCoherentRotation(R2)) {
					cout << "resulting rotation is not coherent\n";
					P1 = 0;
					return false;
				}
					
				P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
				//cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
				reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
					
				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
					//cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
					reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
						
					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						cout << "Shit." << endl; 
						return false;
					}
				}				
			}			
		}
		*/
        //cout << "Found P1: " << endl << Mat(P1) << endl;

		for (unsigned int i=0; i < tstResult[bestI].pcloud.size(); i++)
			outCloud.push_back(tstResult[bestI].pcloud[i]);
	}		
		
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. (" << t <<" sec)"<< endl;

    return true;
}
