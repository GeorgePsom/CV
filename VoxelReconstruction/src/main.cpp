#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CameraCalibration.h"


using namespace nl_uu_science_gmt;

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	CameraCalibration cameraCalibration1("data/cam1/intrinsics.avi", "data/cam1/intrinsics.xml");
	std::cout << "Camera 1 calibration finished" << std::endl;
	CameraCalibration cameraCalibration2("data/cam2/intrinsics.avi", "data/cam2/intrinsics.xml");
	std::cout << "Camera 2 calibration finished" << std::endl;
	CameraCalibration cameraCalibration3("data/cam3/intrinsics.avi", "data/cam3/intrinsics.xml");
	std::cout << "Camera 3 calibration finished" << std::endl;
	CameraCalibration cameraCalibration4("data/cam4/intrinsics.avi", "data/cam4/intrinsics.xml");
	std::cout << "Camera 4 calibration finished" << std::endl;
	waitKey();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	return EXIT_SUCCESS;
}
