/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef VOXELRECONSTRUCTION_H_
#define VOXELRECONSTRUCTION_H_

#include <string>
#include <vector>

#include "controllers/Camera.h"
#include "PolyVoxCore/CubicSurfaceExtractorWithNormals.h"
#include "PolyVoxCore/MarchingCubesSurfaceExtractor.h"
#include "PolyVoxCore/SurfaceMesh.h"
#include "PolyVoxCore/SimpleVolume.h"


namespace nl_uu_science_gmt
{

class VoxelReconstruction
{
	const std::string m_data_path;
	const int m_cam_views_amount;

	std::vector<Camera*> m_cam_views;

public:
	VoxelReconstruction(const std::string &, const int);
	virtual ~VoxelReconstruction();

	static void showKeys();

	void run(int, char**, PolyVox::SimpleVolume<uint8_t>& volData);
};

} /* namespace nl_uu_science_gmt */

#endif /* VOXELRECONSTRUCTION_H_ */
