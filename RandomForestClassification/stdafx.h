// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#include "targetver.h"
#include <stdio.h>
#include <tchar.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <random>
#include <opencv2/opencv.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
using namespace std;
using namespace cv;

#ifdef _DEBUG
//Debugモードの場合
#pragma comment(lib,"opencv_core249d.lib")		// opencv_core
#pragma comment(lib,"opencv_imgproc249d.lib")		// opencv_imgproc
#pragma comment(lib,"opencv_highgui249d.lib")		// opencv_highgui
#pragma comment(lib,"opencv_objdetect249d.lib")	// opencv_objdetect
//以下、必要に応じて追加
#pragma comment(lib,"opencv_ml249d.lib")			// opencv_ml
#pragma comment(lib,"opencv_features2d249d.lib")	// opencv_features2d
#pragma comment(lib,"opencv_video249d.lib")		// opencv_video
#pragma comment(lib,"opencv_calib3d249d.lib")		// opencv_calib3d
#pragma comment(lib,"opencv_flann249d.lib")		// opencv_flann
#pragma comment(lib,"opencv_contrib249d.lib")		// opencv_contrib
#pragma comment(lib,"opencv_legacy249d.lib")		// opencv_legacy
#pragma comment(lib,"opencv_gpu249d.lib")			// opencv_gpu
#else
//Releaseモードの場合
#pragma comment(lib,"opencv_core249.lib")			// opencv_core
#pragma comment(lib,"opencv_imgproc249.lib")		// opencv_imgproc
#pragma comment(lib,"opencv_highgui249.lib")		// opencv_highgui
#pragma comment(lib,"opencv_objdetect249.lib")	// opencv_objdetect
//以下、必要に応じて追加
#pragma comment(lib,"opencv_ml249.lib")			// opencv_ml
#pragma comment(lib,"opencv_features2d249.lib")	// opencv_features2d
#pragma comment(lib,"opencv_video249.lib")		// opencv_video
#pragma comment(lib,"opencv_calib3d249.lib")		// opencv_calib3d
#pragma comment(lib,"opencv_flann249.lib")		// opencv_flann
#pragma comment(lib,"opencv_contrib249.lib")		// opencv_contrib
#pragma comment(lib,"opencv_legacy249.lib") 		// opencv_legacy
#pragma comment(lib,"opencv_gpu249.lib")			// opencv_gpu
#endif

#include "RandomForest.h"