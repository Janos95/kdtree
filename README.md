lightweight kd tree implementation on top of Magnum/Corrade.\
It does not copy you pointcloud, instead you pass (a potentially strided) view
of your data to the tree.\
In my measurements, constructing the kd tree is about 15 percent faster than [nanoflann](https://github.com/jlblancoc/nanoflann)
and a nearest neighbor query is about 10 percent faster.\
In my measurements it is about 10 percent faster\
![Alt Text](https://github.com/Janos95/kdtree/blob/master/kdtree.gif)
