# DiscreteMeshOptimization
Discrete Mesh Optimization implemented in C++ and CUDA. Currently available for flat 2D Meshes. Code for Surface and Volume Meshes will follow as soon as our code is optimized and cleaned-up. We are working on it.


# Setup
Windows:
- Download and install OpenMesh and Eigen.

OpenMesh (7.1): https://www.openmesh.org/

Eigen (3.3.5): http://eigen.tuxfamily.org/index.php?title=Main_Page

Installing will work the same way as setting up DMO itself:
- Download and install CMake (at least Version 3.8)
- Start CMake Gui and choose DMOSmoothing as Source Directory.
- Choose a Build Directory, e.g. create a new Folder "build" in DMOSmoothing.
- Click "add Entry" and enter "CMAKE_INSTALL_PREFIX". Select the path to the installation of OpenMesh and Eigen.
- Click "Configure", once finished click "Generate".
- Open the "DMO.sln"-file in the build directory which will open up Visual Studio.
- Select "DMO" as Startup Project and Run it.