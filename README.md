# ICGproject-watercolorization

This project implemented this paper,

Wang, M., Wang, B., Fei, Y., Qian, K., Wang, W., Chen, J., & Yong, J. (2014). Towards Photo Watercolorization with Artistic Verisimilitude. *IEEE Transactions on Visualization and Computer Graphics, 20*, 1451-1460.

The results mentioned in this paper was not fully reproduced in this project. But we have achieved similar effects and obtained desirable results for some images.

This project was originally written in Arch Linux, with CMake version 3.13.2, GCC version 8.2.1, and OpenCV version 4.0.1.

## Dependencies

- GCC (with C++17 support)
- OpenCV
- CMake

## Building

To build this project, `cd` into the `build` directory and type:

```
cmake ..
make
```

To build the clustering tool, `cd` into the `ClusterTool` directory and type:

```
mkdir build
cd build
cmake ..
make
```

## Usage

To apply watercolorization effect to an image:

```
./Watercolorization /path/to/inputfile [/path/to/outputfile]
```

If `/path/to/outputfile` was not given, the watercolorized image would be output as `output.jpg` in the working directory.

To use the clustering tool, place the training images into `PicutreDatabase/Original` and run:

```
./Clustering ../../PictureDatabase
```

The pictures will be classified into 20 classes. Folders corresponding to each class will be created, with the symbolic links to the original files automatically generated in them. The trained centers of the k-means algorithm will be stored in  `model`, which should be put in the same directory of `Watercolorization` in order to enable the auto-detection of the closest style of images.

## License

This project was licensed under GNU GPL3. Check the `LICENSE` file for more details.
