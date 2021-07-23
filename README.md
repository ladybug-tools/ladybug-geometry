
![Ladybug](http://www.ladybug.tools/assets/img/ladybug.png)

[![Build Status](https://github.com/ladybug-tools/ladybug-geometry/workflows/CI/badge.svg)](https://github.com/ladybug-tools/ladybug-geometry/actions)
[![Coverage Status](https://coveralls.io/repos/github/ladybug-tools/ladybug-geometry/badge.svg?branch=master)](https://coveralls.io/github/ladybug-tools/ladybug-geometry?branch=master)

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)](https://www.python.org/downloads/release/python-270/) [![IronPython](https://img.shields.io/badge/ironpython-2.7-red.svg)](https://github.com/IronLanguages/ironpython2/releases/tag/ipy-2.7.8/)

# ladybug-geometry

Ladybug geometry is a Python library that houses geometry objects used throughout the
Ladybug Tools core libraries.

## Installation

`pip install -U ladybug-geometry`

## [API Documentation](https://www.ladybug.tools/ladybug-geometry/docs/)

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

## Currently Supported Capabilities of this Library

- Vector Math
- Calculate Bounding Box (Min, Max, Center)
- Compute Area + Perimeter of Planar Geometry
- Check Concavity and Clockwise Ordering of 2D Geometry
- Triangulate Planar Geometry
- Compute Triangle + Quad Areas, Centroids, and Normals
- Move Geometry
- Rotate Geometry Around an Axis
- Mirror Geometry
- Scale Geometry from a Base Point
- Is Point Inside 2D Polygon
- 3D Face Intersection with a Ray or Line
- Mesh Grid Generation from a 3D Face
- Windows Based on Ratio with a Face
- Solve Adjacencies
- Generate Louvers, Fins and Overhangs from a Face
- Check if a 3D PolyFace is a Closed Solid
- Ensure All Faces of a Solid PolyFace are Point Outwards
- Join Polylines and Polyfaces
- Check if a Point is Inside a Closed 3D Polyface
- Boolean a Set of 2D Curves (joining the naked edges around them)

## Capabilities that should eventually be a part of this library

- [ ] Create Matching Zone Surfaces (intersection of surfaces with one another). OpenStudio's boost geometry has methods for this [as @saeranv shows here](https://github.com/mostaphaRoudsari/honeybee/issues/700)

## Officially Unsupported Capabilities for which One Must Rely on CAD Interfaces

- Conversion of Curved Surfaces to Planar Surfaces (including both single curvature and double curvature)
- Fancier Meshing (eg. gridded meshing that completely fills the base surface)
- Solid Boolean Unions (this should not be needed for anything in Ladybug Tools)

## Reasons for this Library

We initially debated whether geometry computation should be placed largely on the CAD plugins or
whether it should be included in the core.  As we developed the core libraries out, it became clear
that there are large advantages to having it in the core (ie. cross compatibility between
the CAD plugins, ability to process more inputs from command line, and shear speed
since the CAD libraries are made to address many more geometric use cases than are typically needed).
So we have decided to include geometry computation as part of the Ladybug Tools core.

We looked into using other geometry computation libraries for the core including:

- [Rhino3dm](https://github.com/mcneel/rhino3dm)
- [Blender API (bpy)](https://docs.blender.org/api/current/)
- [Topologic](https://topologic.app/Software/)

However, Rhino3dm lacks basic types of computation that is needed in the core (like generating a
grid of points from a surface).
Furthermore, Blender library only works in Python3 and this would break our workflows for the
Grasshopper and Dynamo plugins, where rely on IronPython.
Topologic seems to have many things that we need but it appears that it has C dependencies, making
it unusable from IronPython.  Furthermore, its dual license may create some difficulties for certain
use cases of Ladybug Tools.

After considering it further, we realized that many of the calculations that we need can be done
fairly easily as long as the geometry is planar.  Since all of the geometry going to the engines (Radiance, E+)
is eventually converted to a planar format anyway, we made the decision that the core libraries will support
certain basic types of geometry computation for planar objects only.  We planned to do this by taking the
most relevant parts of existing open source geometry libraries, including [euclid](https://pypi.org/project/euclid/)
and OpenStudio. Thus this repository was born!
