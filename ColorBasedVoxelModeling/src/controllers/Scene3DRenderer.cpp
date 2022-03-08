/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;


	createFloorGrid();
	setTopView();

	calcThresholds(m_cameras[1]);

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);
		processForeground(m_cameras[c]);
	}
	return true;
}

void Scene3DRenderer::processBackground()
{

}


void Scene3DRenderer::calcThresholds(
	Camera* camera)
{
	Mat hsv_image;


	cvtColor(camera->getVideoFrame(0), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	int pixelNum = camera->getVideoFrame(0).size().width * camera->getVideoFrame(0).size().height;

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	Mat hsv_mask;
	cvtColor(camera->getMask(), hsv_mask, CV_BGR2HSV);
	Mat hsv_mask_dilated;
	dilate(hsv_mask, hsv_mask_dilated, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)), Point(-1, 1), 5);
	vector<Mat> mask_channels;
	split(hsv_mask_dilated, mask_channels);


	int max = 0;
	for (int v = 0; v < 255; v++)
	{
		Mat tmp, foreground, background;
		absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
		threshold(tmp, foreground, v, 255, CV_THRESH_BINARY);

		bitwise_xor(foreground, mask_channels[2], foreground);

		Mat result = imgProcPipeline(hsv_image, foreground);
		foreground = result;

		int nCorrectPixels = pixelNum - countNonZero(foreground);
		if (nCorrectPixels > max)
		{
			max = nCorrectPixels;
			m_v_threshold = v;
			m_pv_threshold = v;

		}
	}

	Mat tmp, foregroundV;
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, foregroundV, m_v_threshold, 255, CV_THRESH_BINARY);

	max = 0;
	for (int s = 0; s < 255; s++)
	{
		Mat tmp, foreground, background;
		absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
		threshold(tmp, foreground, s, 255, CV_THRESH_BINARY);
		bitwise_xor(foreground, mask_channels[1], foreground);

		Mat result = imgProcPipeline(hsv_image, foreground);
		foreground = result;

		int nCorrectPixels = pixelNum - countNonZero(foreground);
		if (nCorrectPixels > max)
		{
			max = nCorrectPixels;
			m_s_threshold = s;
			m_ps_threshold = s;
		}
	}

	Mat foregroundS;
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, foregroundS, m_v_threshold, 255, CV_THRESH_BINARY);


	max = 0;
	for (int h = 0; h < 255; h++)
	{
		Mat tmp, foreground, background;
		absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
		threshold(tmp, foreground, h, 255, CV_THRESH_BINARY);
		
		bitwise_xor(foreground, mask_channels[0], foreground);

		Mat result = imgProcPipeline(hsv_image, foreground);
		foreground = result;

		int nCorrectPixels = pixelNum - countNonZero(foreground);
		if (nCorrectPixels > max)
		{
			max = nCorrectPixels;
			m_h_threshold = h;
			m_ph_threshold = h;
		}
	}
}

/**
 * Image Processing pipeline
 */
Mat Scene3DRenderer::imgProcPipeline(Mat hsv_image, Mat foreground)
{

	Mat greenMask;
	inRange(hsv_image, Scalar(60, 100, 80), Scalar(88, 255, 150), greenMask);
	bitwise_not(greenMask, greenMask);

	bitwise_and(foreground, greenMask, foreground);

	erode(foreground, foreground, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)), Point(-1, 1));

	dilate(foreground, foreground, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)), Point(-1, 1));
	dilate(foreground, foreground, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)), Point(-1, 1));

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	int contourThres = 100;

	findContours(foreground, contours, hierarchy,
		RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if (!contours.empty()) {
		int maxSize = 0;

		int idx = 0, maxInd = 0;

		for (idx = 0; idx < contours.size(); idx++)
		{
			/*if (contours[idx].size() > maxSize)
			{
				maxSize = contours[idx].size();
				maxInd = idx;
			}*/

			if (contours[idx].size() > contourThres)
			{
				drawContours(foreground, contours, idx, 255, FILLED);
			}

		}

	}

	vector<vector<Point> > contoursNew;
	vector<Vec4i> hierarchyNew;

	findContours(foreground, contoursNew, hierarchyNew,
		RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if (!contoursNew.empty()) {
		int maxSize = 0;

		int idx = 0, maxInd = 0;

		for (idx = 0; idx < contoursNew.size(); idx++)
		{

			if (contoursNew[idx].size() > contourThres)
			{
				drawContours(foreground, contoursNew, idx, 255, FILLED);
			}
			else
			{
				drawContours(foreground, contoursNew, idx, 0, FILLED);
			}

		}
	}

	contoursNew.clear();

	dilate(foreground, foreground, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)), Point(-1, 1));

	findContours(foreground, contoursNew, hierarchyNew,
		RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if (!contoursNew.empty()) {
		int maxSize = 0;

		int idx = 0, maxInd = 0;

		for (idx = 0; idx < contoursNew.size(); idx++)
		{

			if (contoursNew[idx].size() > contourThres)
			{
				drawContours(foreground, contoursNew, idx, 255, FILLED);
			}
		}
	}

	inRange(hsv_image, Scalar(55, 100, 80), Scalar(85, 255, 200), greenMask);
	bitwise_not(greenMask, greenMask);
	bitwise_and(foreground, greenMask, foreground);


	erode(foreground, foreground, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)), Point(-1, 1));

	return foreground;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
	Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	Mat tmp, foreground, background;
	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	threshold(tmp, foreground, m_h_threshold, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, background, m_s_threshold, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);


	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, background, m_v_threshold, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);
	Mat result;

	result = imgProcPipeline(hsv_image, foreground);

	camera->setForegroundImage(result);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
