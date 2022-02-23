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
	CameraCalibration cameraCalibration("data/cam1/intrinsics.avi", "data/cam1/intrinsics.xml");
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	return EXIT_SUCCESS;
}
