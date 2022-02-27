#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CameraCalibration.h"

#include "PolyVoxCore/CubicSurfaceExtractorWithNormals.h"
#include "PolyVoxCore/MarchingCubesSurfaceExtractor.h"
#include "PolyVoxCore/SurfaceMesh.h"
#include "PolyVoxCore/SimpleVolume.h"



using namespace nl_uu_science_gmt;
using namespace PolyVox;




void createSphereInVolume(SimpleVolume<uint8_t>& volData, float fRadius)
{
    //This vector hold the position of the center of the volume
    Vector3DFloat v3dVolCenter(volData.getWidth() / 2, volData.getHeight() / 2, volData.getDepth() / 2);

    //This three-level for loop iterates over every voxel in the volume
    for (int z = 0; z < volData.getDepth(); z++)
    {
        for (int y = 0; y < volData.getHeight(); y++)
        {
            for (int x = 0; x < volData.getWidth(); x++)
            {
                //Store our current position as a vector...
                Vector3DFloat v3dCurrentPos(x, y, z);
                //And compute how far the current position is from the center of the volume
                float fDistToCenter = (v3dCurrentPos - v3dVolCenter).length();

                uint8_t uVoxelValue = 0;

                //If the current voxel is less than 'radius' units from the center then we make it solid.
                if (fDistToCenter <= fRadius)
                {
                    //Our new voxel value
                    uVoxelValue = 255;
                }

                //Wrte the voxel value into the volume
                volData.setVoxelAt(x, y, z, uVoxelValue);
            }
        }
    }
}

int main(
		int argc, char** argv)
{
	SimpleVolume<uint8_t> volData(PolyVox::Region(Vector3DInt32(0, 0, 0), Vector3DInt32(127, 127, 127)));
    createSphereInVolume(volData, 100.0f);

    SurfaceMesh<PositionMaterialNormal> mesh;
    CubicSurfaceExtractorWithNormals< SimpleVolume<uint8_t> > surfaceExtractor(&volData, volData.getEnclosingRegion(), &mesh);
    surfaceExtractor.execute();
    std::cout << "Marching cubes completed! " << std::endl;
	VoxelReconstruction::showKeys();
	// Camera calibration is done once and stored in intrinsics.xml. Uncomment if you wish to run calibration again.
	/*CameraCalibration cameraCalibration1("data/cam1/intrinsics.avi", "data/cam1/intrinsics.xml");
	std::cout << "Camera 1 calibration finished" << std::endl;
	CameraCalibration cameraCalibration2("data/cam2/intrinsics.avi", "data/cam2/intrinsics.xml");
	std::cout << "Camera 2 calibration finished" << std::endl;
	CameraCalibration cameraCalibration3("data/cam3/intrinsics.avi", "data/cam3/intrinsics.xml");
	std::cout << "Camera 3 calibration finished" << std::endl;
	CameraCalibration cameraCalibration4("data/cam4/intrinsics.avi", "data/cam4/intrinsics.xml");
	std::cout << "Camera 4 calibration finished" << std::endl;
	waitKey();*/
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv, mesh);

	return EXIT_SUCCESS;
}
