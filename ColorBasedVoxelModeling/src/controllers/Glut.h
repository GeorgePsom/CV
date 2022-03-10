/*
 * Glut.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef GLUT_H_
#define GLUT_H_

#ifdef _WIN32
#include <Windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __linux__
#include <GL/glut.h>
#include <GL/glu.h>
#endif

// i am not sure about the compatibility with this...
#define MOUSE_WHEEL_UP   3
#define MOUSE_WHEEL_DOWN 4
#include "PolyVoxCore/CubicSurfaceExtractorWithNormals.h"
#include "PolyVoxCore/MarchingCubesSurfaceExtractor.h"
#include "PolyVoxCore/SurfaceMesh.h"
#include "PolyVoxCore/SimpleVolume.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <vector>

namespace nl_uu_science_gmt
{

class Scene3DRenderer;

class Glut
{
	Scene3DRenderer &m_scene3d;
	/*PolyVox::SurfaceMesh<PolyVox::PositionMaterialNormal> mesh;*/
	PolyVox::SimpleVolume<uint8_t>* volData;
	int dispState = 0; // 0 for voxels, 1 for colored voxels filled with black, 2 for all colored voxels, 3 for mesh

	static Glut* m_Glut;

	static void drawGrdGrid();
	static void drawCamCoord();
	static void drawVolume();
	static void drawArcball();
	static void drawVoxels();
	static void drawWCoord();
	static void drawInfo();
	static void drawMesh();

	static void matchColorInds(std::vector<std::vector<cv::Point3f>>& colorsAvg, std::vector<int>& inds, int clustSize);

	static void findColModel(std::vector<int>& clusterIndices, std::vector<cv::Point2f>& clusterCenters,
		std::vector<std::vector<cv::Point3f>>& avgColors, bool offline);

	static inline void perspectiveGL(
			GLdouble, GLdouble, GLdouble, GLdouble);

#ifdef _WIN32
	static void SetupPixelFormat(HDC hDC);
	static LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
#endif

public:
	Glut(
			Scene3DRenderer &, PolyVox::SimpleVolume<uint8_t>& );
	virtual ~Glut();

#ifdef __linux__
	void initializeLinux(
			const char*, int, char**);
	static void mouse(
			int, int, int, int);
#endif
#ifdef _WIN32
	int initializeWindows(const char*);
	void mainLoopWindows();
#endif

	static void keyboard(
			unsigned char, int, int);
	static void motion(
			int, int);
	static void reshape(
			int, int);
	static void reset();
	static void idle();
	static void display();
	static void update(
			int);
	static void quit();

	Scene3DRenderer& getScene3d() const
	{
		return m_scene3d;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* GLUT_H_ */
