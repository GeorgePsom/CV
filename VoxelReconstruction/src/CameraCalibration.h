#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;
class Settings
{
public:
    Size boardSize = Size(8, 6);              // The size of the board -> Number of items by width and height
    float squareSize = 115.0f;            // The size of a square in your defined unit (point, millimeter,etc).
    int nrFrames;                // The number of frames to use from the input for calibration
    float aspectRatio;           // The aspect ratio
    int delay;                   // In case of a video input
    bool writePoints;            // Write detected feature points
    bool writeExtrinsics;        // Write extrinsic parameters
    bool writeGrid;              // Write refined 3D target grid points
    bool calibZeroTangentDist;   // Assume zero tangential distortion
    bool calibFixPrincipalPoint; // Fix the principal point at the center
    bool flipVertical;           // Flip the captured images around the horizontal axis
    string outputFileName;       // The name of the file where to write
    bool showUndistorted;        // Show undistorted images after calibration
    string input;                // The input ->
    bool useFisheye;             // use fisheye camera model for calibration
    bool fixK1;                  // fix K1 distortion coefficient
    bool fixK2;                  // fix K2 distortion coefficient
    bool fixK3;                  // fix K3 distortion coefficient
    bool fixK4;                  // fix K4 distortion coefficient
    bool fixK5;                  // fix K5 distortion coefficient

    int cameraID;
    vector<string> imageList;
    size_t atImageList;
    VideoCapture inputCapture;
    bool goodInput;
    int flag;

    Settings(const std::string& videoFile)
    {
        flipVertical = false;
        nrFrames = 60;
        fixK1 = fixK2 = fixK3 = false;
        fixK4 = fixK5 = true;
        calibFixPrincipalPoint = false;
        calibZeroTangentDist = true;
        aspectRatio = 0.0f;
        input = videoFile;
        inputCapture.open(input);

        flag = 0;
        if (calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
        if (calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
        if (aspectRatio)            flag |= CALIB_FIX_ASPECT_RATIO;
        if (fixK1)                  flag |= CALIB_FIX_K1;
        if (fixK2)                  flag |= CALIB_FIX_K2;
        if (fixK3)                  flag |= CALIB_FIX_K3;
        if (fixK4)                  flag |= CALIB_FIX_K4;
        if (fixK5)                  flag |= CALIB_FIX_K5;
        atImageList = 0;


    }

private:
    string patternToUse;
};
class CameraCalibration
{
public:
    CameraCalibration(const std::string& videoPath);


private: 
    Mat nextImage();
    bool runCalibrationAndSave(Settings& s, Size imageSize, vector<vector<Point2f>> imagePoints, float grid_width);
    bool  runCalibration(Settings& s, Size& imageSize, vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
        vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints,
        float grid_width);

    double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs);

    void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
public:
    Mat cameraMatrix, distCoeffs;
private:
    Settings s;

    

};

