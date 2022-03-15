# OUTDATED
There is a new version of this code that is based on templates. It is faster, shorter and works on both, GPU and CPU. Additionally, it is much easier to add new quality metrics:
https://github.com/DanielZint/dmo_templated

# DiscreteMeshOptimization
Discrete Mesh Optimization implemented in C++ and CUDA. Currently available for flat 2D Meshes. Code for Surface and Volume Meshes will follow as soon as our code is optimized and cleaned-up. We are working on it.

# Prerequisites
CUDA, OpenMesh (7.1) and Eigen (3.3.5) have to be installed to compile this project.
- OpenMesh: https://www.openmesh.org/
- Eigen: http://eigen.tuxfamily.org/

# Setup
__Windows:__
- Download and install CMake (at least Version 3.8)
- Start CMake Gui and choose DiscreteMeshOptimization as Source Directory.
- Choose a Build Directory, e.g. create a new Folder "build" in DiscreteMeshOptimization.
- Click "add Entry" and enter "CMAKE_INSTALL_PREFIX". Select the path to the installation of OpenMesh and Eigen if they have not been installed to the default location.
- Click "Configure", once finished click "Generate".
- Open the "DMO.sln"-file in the build directory which will open up Visual Studio.
- Select "DMO" as Startup Project and Build it.

__Linux:__
- Create a build folder in the root directory of the project.
- Execute "cmake .. -DCMAKE_INSTALL_PREFIX=x" inside the build folder where x is the location of the installation of OpenMesh and Eigen.
- call make.
