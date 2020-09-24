# artery_vision

## Goal
Correlate X-ray images from two cameras to create 3d model of brain arteries, localisation of catheter and motion planning for use in brain surgery.

## Components
#Data Folder: 
Everything you need to create sample data to process. Blender Scenes, render scripts and rendered imagery, together with meta files about location and properties of cameras. To use it you need to have blender installed, and you might need to adjust the paths to blender and to your local output folder in the scripts.

# Include: 
Header files for libraries. 

arteryNet.h provides the graph, and allows the graph to be exported for rendering. 

camera.h provides knowledge about camera properties. Enables you to shoot rays through pixels, and correlate this with a ray from another camera. Also provides functions needed to render on to the camera canvas, for visualization in the app. 

imageData.h is the central library that manages loading, storing and processing of image data. It builds the graph based on x-ray imagery, using the camera meta data.

Eigen: for linear algebra.

OpenCV (no build included): Must be built with QT enabled, tested with version 4.4

# Test:
Some simple unit test used during developement of the libraries, rather uncomplete and with low coverage.

# Source:
Source files including different apps showing the use of the libraries. Also the source files of the libraries.

## Build instructions
Requires OpenCV to be installed, built with QT (for the apps). CMake files should be self explanatory and work out of the box, tested on Ubuntu 18.
There are different targets for the libraries, the tests, and the actual apps.
