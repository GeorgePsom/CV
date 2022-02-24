#include "CameraCalibration.h"

CameraCalibration::CameraCalibration(const std::string& videoPath, const std::string& outPath) :
    s(videoPath)
{
    vector<vector<Point2f> > imagePoints;
    int grid_width = s.squareSize * (s.boardSize.width - 1);
    namedWindow("Test", WINDOW_AUTOSIZE);
    Mat view = nextImage();;
    Size imageSize = Size(0,0);
    int counter = 1;
    for (;;)
    {
        
        char key = waitKey(1000.0f / 900.0f);
       
       
        if (imageSize.width == 0)
            imageSize = view.size();

        if (view.empty())
        {
            std::cout << "Nr of frames: " << imagePoints.size() << std::endl;
            runCalibrationAndSave(s, imageSize, imagePoints, grid_width, outPath);
            break;
        }


        vector<Point2f> pointBuf;
        bool found; 
        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH 
            | CALIB_CB_NORMALIZE_IMAGE;
        found = findChessboardCorners(view, s.boardSize, pointBuf, chessBoardFlags);

        if (found)
        {
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointBuf, Size(11, 11),
                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

            drawChessboardCorners(view, s.boardSize, Mat(pointBuf), found);
            imagePoints.push_back(pointBuf);
        }
        if (found)
        {
            while (counter < 15)
            {
                view = nextImage();
                counter++;

            }
            counter = counter == 15 ? 1 : counter;
        }
        else
            view = nextImage();
       
        
    }
}

Mat CameraCalibration::nextImage()
{
    Mat result;
    if (s.inputCapture.isOpened())
    {
        Mat view0;
        s.inputCapture >> view0;
        view0.copyTo(result);
    }
    else if (s.atImageList < s.imageList.size())
        result = imread(s.imageList[s.atImageList++], IMREAD_COLOR);

    return result;
}

bool CameraCalibration::runCalibrationAndSave(Settings& s, Size imageSize, vector<vector<Point2f>> imagePoints, float grid_width, std::string outPath)
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;
    vector<Point3f> newObjPoints;

    bool ok = runCalibration(s, imageSize, imagePoints, rvecs, tvecs, reprojErrs,
        totalAvgErr, newObjPoints, grid_width);
    std::cout << (ok ? "Calibration succeeded" : "Calibration failed")
        << ". avg re projection error = " << totalAvgErr << endl;



    if (ok)
        SaveCameraParams(outPath);
    return ok;
}

bool CameraCalibration::SaveCameraParams(std::string outPath)
{

    FileStorage fs(outPath, FileStorage::WRITE);


    fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
    fs.release();

    return 0;
}


void CameraCalibration::calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();

   
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
       
    
}

bool CameraCalibration::runCalibration(Settings& s, Size& imageSize, vector<vector<Point2f>> imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs, vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints, float grid_width)
{
    //! [fixed_aspect]
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (s.flag & CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = s.aspectRatio;
    ////! [fixed_aspect]
    //if (s.useFisheye) {
    //    distCoeffs = Mat::zeros(4, 1, CV_64F);
    //}
    //else {
    distCoeffs = Mat::zeros(8, 1, CV_64F);
   
    // First camera calibration from all the images
    vector<vector<Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0]);
    objectPoints[0][s.boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
    newObjPoints = objectPoints[0];
    vector<Point3f> newObjPointsTemp = objectPoints[0];
    int iter = 0;

    // optimalImagePoints holds the current images. This vector will be reduced in every iteration
    vector<vector<Point2f>> optimalImagePoints(imagePoints.size());
    for (int i = 0; i < imagePoints.size(); i++)
    {
        optimalImagePoints[i] = imagePoints[i];
    }

    bool ok;



    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms;


    if (s.useFisheye) {
        Mat _rvecs, _tvecs;
        rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, _rvecs,
            _tvecs, s.flag);

        rvecs.reserve(_rvecs.rows);
        tvecs.reserve(_tvecs.rows);
        for (int i = 0; i < int(objectPoints.size()); i++) {
            rvecs.push_back(_rvecs.row(i));
            tvecs.push_back(_tvecs.row(i));
        }
    }
    else {
        int iFixedPoint = -1;
        /*if (release_object)
            iFixedPoint = s.boardSize.width - 1;*/

        rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
            cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
            s.flag | CALIB_USE_LU);
    }

   /* if (release_object) {
        cout << "New board corners: " << endl;
        cout << newObjPoints[0] << endl;
        cout << newObjPoints[s.boardSize.width - 1] << endl;
        cout << newObjPoints[s.boardSize.width * (s.boardSize.height - 1)] << endl;
        cout << newObjPoints.back() << endl;
    }*/

    cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

    ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs);
    cout << "Re-projection error reported by calibrateCamera: " << totalAvgErr << endl;
    // Initialization for the needed variables
    float maxError = totalAvgErr;
    Mat bestCameraMatrix = cameraMatrix;
    Mat bestDistCoeffs = distCoeffs;
    vector<Mat> bestRvecs = rvecs;
    vector<Mat> bestTvecs = tvecs;

    for (;;)
    {
        break;
        float bestError = 1000.0f;
        int bestCandidate = -1;
        // Iterates over all current images to find the worst candidate (the image that if removed will produce the least error)
        for (int j = 0; j < optimalImagePoints.size(); j++)
        {
            int imagePointsSize = optimalImagePoints.size() - 1;
            vector<vector<Point2f>> imagePointsMinusOne;
            for (int i = 0; i < optimalImagePoints.size(); i++)
            {
                if (i == j)
                    continue;
                imagePointsMinusOne.push_back(optimalImagePoints[i]);
            }
            objectPoints.resize(imagePointsSize, newObjPoints);

            //Find intrinsic and extrinsic camera parameters



            if (s.useFisheye) {
                Mat _rvecs, _tvecs;
                rms = fisheye::calibrate(objectPoints, imagePointsMinusOne, imageSize, cameraMatrix, distCoeffs, _rvecs,
                    _tvecs, s.flag);

                rvecs.reserve(_rvecs.rows);
                tvecs.reserve(_tvecs.rows);
                for (int i = 0; i < int(objectPoints.size()); i++) {
                    rvecs.push_back(_rvecs.row(i));
                    tvecs.push_back(_tvecs.row(i));
                }
            }
            else {
                int iFixedPoint = -1;
                /*if (release_object)
                    iFixedPoint = s.boardSize.width - 1;*/

                rms = calibrateCameraRO(objectPoints, imagePointsMinusOne, imageSize, iFixedPoint,
                    cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
                    s.flag | CALIB_USE_LU);
            }

            /*if (release_object) {
                cout << "New board corners: " << endl;
                cout << newObjPoints[0] << endl;
                cout << newObjPoints[s.boardSize.width - 1] << endl;
                cout << newObjPoints[s.boardSize.width * (s.boardSize.height - 1)] << endl;
                cout << newObjPoints.back() << endl;
            }*/

            //cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

            ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

            objectPoints.clear();
            objectPoints.resize(imagePointsSize, newObjPoints);
            totalAvgErr = computeReprojectionErrors(objectPoints, imagePointsMinusOne, rvecs, tvecs);
            // Store if is the best so far
            if (totalAvgErr < bestError)
            {
                bestCameraMatrix = cameraMatrix;
                bestDistCoeffs = distCoeffs;
                bestRvecs = rvecs;
                bestTvecs = tvecs;
                bestError = totalAvgErr;
                bestCandidate = j;
            }
        }

        // After we find the image to be removed, we update the vector that holds the rest of the images
        vector<vector<Point2f>> temp;
        for (int i = 0; i < optimalImagePoints.size(); i++)
        {
            temp.push_back(optimalImagePoints[i]);
        }
        optimalImagePoints.clear();
        //optimalImagePoints.resize(temp.size() - 1);
        for (int i = 0; i < temp.size(); i++)
        {
            if (i == bestCandidate)
                continue;
            optimalImagePoints.push_back(temp[i]);
        }




        // Loop that rejects images ends, when we reach the optimal error, or the error does not improve beyond an epsilon value, or we discard half the images.
        if (bestError > maxError || abs(bestError - maxError) < 0.001f || optimalImagePoints.size() == imagePoints.size() / 2)
            break;
        else
        {
            maxError = bestError;
            cout << "Iteration:  " << ++iter << ". Re-projection error reported by calibrateCamera: " << maxError << endl;
        }


    }
    cameraMatrix = bestCameraMatrix;
    distCoeffs = bestDistCoeffs;
    rvecs = bestRvecs;
    tvecs = bestTvecs;

    return ok;
}


double CameraCalibration::computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs)
{
    vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
       
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
       
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}
