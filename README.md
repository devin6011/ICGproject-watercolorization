# ICGproject-watercolorization

This project tries to implement this paper in C++17 and OpenCV 4.0:

Wang, M., Wang, B., Fei, Y., Qian, K., Wang, W., Chen, J., & Yong, J. (2014). Towards Photo Watercolorization with Artistic Verisimilitude. *IEEE Transactions on Visualization and Computer Graphics, 20*, 1451-1460.

This project is originally written in Arch Linux, with CMake version 3.13.2, GCC version 8.2.1, and OpenCV version 4.0.1.

To build this project, `cd` into the `build` directory and type:

`cmake ..`

`make`

To build the clustering tool, `cd` into the `ClusterTool` directory and type:

`mkdir build`

`cd build`

`cmake ..`

`make`

The usage of Watercolorization is:

`./Watercolorization /path/to/inputfile [/path/to/outputfile]`

If `/path/to/outputfile` is not given, the output file will be stored as `output.jpg` in the same directory.

To use the clustering tool, put the image files for training into `PicutreDatabase/Original` and run:

`./Clustering ../../PictureDatabase`

After that, the pictures will be automatically classified into 20 classes. Each class will have its own folder, and the symbolic links to the original file will be created in it. The trained centers of Kmeans algorithm will also be output as a file named `model`. One should put this model file in the same directory of `Watercolorization` in order to auto-detect the closest style of the image.

This project is licensed under GPL3. You should check the `LICENSE` file for more details.