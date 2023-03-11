
![Ladybug](http://www.ladybug.tools/assets/img/ladybug.png)

[![Build Status](https://github.com/ladybug-tools/ladybug-geometry/workflows/CI/badge.svg)](https://github.com/ladybug-tools/ladybug-geometry/actions)
[![Coverage Status](https://coveralls.io/repos/github/ladybug-tools/ladybug-geometry/badge.svg?branch=master)](https://coveralls.io/github/ladybug-tools/ladybug-geometry?branch=master)

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)](https://www.python.org/downloads/release/python-270/) [![IronPython](https://img.shields.io/badge/ironpython-2.7-red.svg)](https://github.com/IronLanguages/ironpython2/releases/tag/ipy-2.7.8/)

# ladybug-geometry

Ladybug geometry is a Python library that houses geometry objects and geometry
computations methods used throughout the Ladybug Tools core libraries.

The library is designed to work with a wide range of Python environments and
it returns consistent results between them (cPython 2 and 3, IronPython 2).

## Installation

`pip install -U ladybug-geometry`

## [API Documentation](https://www.ladybug.tools/ladybug-geometry/docs/)

## Currently Supported Capabilities of this Library

- Perform Vector Math (Dot, Cross, Angle, Normalize)
- Calculate Bounding Box for any Geometry (Min, Max, Center)
- Subdivide Lines and Arcs
- Compute Perimeter and Area of Planar Geometry
- Check Concavity and Clockwise Ordering of 2D Geometry
- Triangulate Planar Geometry
- Compute Mesh Face Areas, Centroids, and Normals
- Move Any Geometry
- Rotate Any Geometry Around an Axis
- Mirror (Reflect) Any Geometry Over a Plane
- Scale Any Geometry from a Base Point
- Check if a 2D Point Inside 2D Polygon
- Compute [Pole of Inaccessibility](https://en.wikipedia.org/wiki/Pole_of_inaccessibility) for any 2D Polygon
- Perform 2D Polygon Boolean Operations (Union, Intersection, Difference)
- Intersect Colinear 2D Polygon Segments with one Another (for matching lengths)
- Join Line Segments into Polylines
- Calculate 3D Face Plane and Normal from Vertices
- Compute 3D Face Intersection with a Ray or Line
- Generate a Quad Mesh Grid from a 3D Face
- Generate Sub-faces Based on Ratio with a Face (used for window generation)
- Generate Contours and Contour Fins from a Face (used to generate louvers, fins and overhangs)
- Split 3D Coplanar Faces with one Another (for matching areas)
- Solve Adjacencies by Matching 3D Face Geometries
- Join 3D Faces into 3D Polyfaces
- Check if a 3D PolyFace is a Closed Solid
- Ensure All Faces of a Solid 3D PolyFace Point Outwards
- Compute the Volume of a Closed 3D Polyface
- Check if a Point is Inside a Closed 3D Polyface

## Officially Unsupported Capabilities for which One Must Rely on CAD Interfaces

- Conversion of Curved 3D Surfaces to Planar 3D Faces
- Fancier Meshing (eg. gridded meshing that completely fills the base surface)
- Solid Boolean Unions (this should not be needed for anything in Ladybug Tools)

## Acknowledgements

This library was built by combining capabilities of several different open-source
(MIT Licensed) projects, establishing a set of standardized geometry objects that
allowed them all to talk to one another, and adding several other capabilities
with new code. We as a community owe a huge amount of thanks to the open source projects
that provided many of the starting capabilities of this package and we are indebted
to the developers who made their work available under an MIT license for the betterment
of geometry computation everywhere. Where possible, you will find detailed lists of
references in the docstrings of this package's source code. A summary of the key
sources that were used to build this library are as follows:

- [euclid](https://pypi.org/project/euclid/)
- [earcut](https://github.com/mapbox/earcut) and [earcut-python](https://github.com/joshuaskelly/earcut-python)
- [polybooljs](https://github.com/velipso/polybooljs) and [pypolybool](https://github.com/KaivnD/pypolybool)
- [polylabel](https://github.com/Twista/python-polylabel)
- [pySTL](https://github.com/proverbialsunrise/pySTL)
- A countless number of [StackOverflow](https://stackoverflow.com/) experts who answered various geometry questions
- A countless number of [Wikipedia](https://www.wikipedia.org/) authors who described various geometry algorithms

## Reasons for this Library

We initially debated whether the burden of geometry computation should be placed largely
on the CAD environments in which Ladybug Tools operates or whether it should be included
in a dedicated core Python library like this one.

As we developed the core libraries, it became clear that there are large advantages
to having it in the core including:

1. Standardized compatibility of geometry between different CAD plugins (eg. Rhino, Revit), simulation engines (eg. E+, Radiance), and file formats (eg. gbXML, GEM).
2. The ability to perform geometry operations from the core library CLI without the need for CAD software.
3. Improved performance (since a dedicated library could be tailored to the use cases of Ladybug Tools).
4. Reliability and maintain-ability in the face of changes to CAD environments and changing Python conventions.

Items 1 and 4 above proved to be particularly important and so the decision was made
that the Ladybug Tools core libraries would have its own geometry library that was
distinct from CAD plugins.

Before committing to write our own library, we looked into using or tweaking other
comprehensive open source geometry libraries for the core including:

- [Rhino3dm](https://github.com/mcneel/rhino3dm)
- [Blender API (bpy)](https://docs.blender.org/api/current/)
- [Boost Geometry](https://www.boost.org/doc/libs/1_78_0/libs/geometry/doc/html/index.html)
- [Topologic](https://topologic.app/Software/)

However, Rhino3dm lacks basic geometry computation. The Blender library had many capabilities
but it only works in Python3 and this could break certain CAD workflows that rely on
IronPython. Boost Geometry (the geometry library used by the OpenStudio SDK) also had a
lot of functionality but it clearly has C dependencies, making it unusable from IronPython.
Topologic also appeared to have C dependencies, though the most relevant issue was
that its dual license could create challenges for certain use cases of Ladybug Tools.

After considering the situation further, we realized that many of the capabilities that
we needed could be achieved by building off the work of various open source MIT-licensed
projects as long as we committed to using planar geometry. Since all of the geometry
ultimately going to the engines (Radiance, E+) is planar, we made the decision that
the core libraries will primarily support planar objects with no NURBS support and
very limited support for Arcs, Circles, Spheres, Cylinders and Cones.

Thus this repository was born!

## Local Development

1. Clone this repo locally
```console
git clone git@github.com:ladybug-tools/ladybug-geometry.git

# or

git clone https://github.com/ladybug-tools/ladybug-geometry.git
```

2. Install dependencies:
```console
cd ladybug-geometry
pip install -r dev-requirements.txt
pip install -r requirements.txt
```

3. Run Tests:
```console
python -m pytests tests/
```

4. Generate Documentation:
```console
sphinx-apidoc -f -e -d 4 -o ./docs ./ladybug_geometry
sphinx-build -b html ./docs ./docs/_build/docs
```
