
![Ladybug](http://www.ladybug.tools/assets/img/ladybug.png)


[![Build Status](https://travis-ci.org/ladybug-tools/ladybug.svg?branch=master)](https://travis-ci.org/ladybug-tools/ladybug-geometry)
[![Coverage Status](https://coveralls.io/repos/github/ladybug-tools/ladybug-geometry/badge.svg?branch=master)](https://coveralls.io/github/ladybug-tools/ladybug-geometry?branch=master)

[![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)](https://www.python.org/downloads/release/python-270/) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![IronPython](https://img.shields.io/badge/ironpython-2.7-red.svg)](https://github.com/IronLanguages/ironpython2/releases/tag/ipy-2.7.8/)

# ladybug-geometry
Ladybug geometry is a Python library that houses geometry objects used throughout the Ladybug Tools core libraries.

## [API Documentation](https://www.ladybug.tools/ladybug-geometry/docs/ladybug_geometry.html)

## Reasons for this Library
We initially debated whether geometry computation should be placed largely on the CAD plugins or
whether it should be inincluded in the core.  As we developed the core libraries out, it became clear
that there are large advantages to having it in the core (ie. cross compatability between
the CAD plugins, ability to process more inputs from command line, and shear speed
since the CAD libraries are made to address many more geometric use cases than are typically needed).
So we have decided to include geomtery computation as part of the Ladybug Tools core.

We looked into using other geometry computation libraries for the core including:
- [Rhino3dm](https://github.com/mcneel/rhino3dm)
- [Blender API (bpy)](https://docs.blender.org/api/current/)
- [Topologic](https://topologic.app/Software/)

However, Rhino3dm lacks basic types of computation that is needed in the core (like generating a
grid of points from a surface).
Furthermore, Blender library only works in Python3 and this would break our workflows for the
Grasshopper and Dynamo plugins, where rely on IronPython.
Topologic seems to have many things that we need but it appears that it has C dependencies, making
it unusable from ironpython.  Furthermore, its dual license may create some difficulties for certain
use cases of Ladybug Tools.

After considering it further, we realized that many of the calculations that we need can be done
fairly easily as long as the geometry is planar.  Since all of the geometry going to the engines (Radiance, E+)
is eventually converted to a planar format anyway, we made the decision that the core libraries will support
certain basic types of geometry computation for planar objects only.  We planned to do this by taking the
most relevant parts of existing open source geometry libraries, including [euclid](https://pypi.org/project/euclid/)
and OpenStudio. Thus this repository was born!

For this library, we can borrow some of the math from the previous open source libaraies
listed above (Rhino3dm and Blender), as well as other projects like
[this PhD on Grid Generation for Radiance](https://www.radiance-online.org/community/workshops/2015-philadelphia/presentations/day1/STADICUtilities-Radiance%20Workshop2015.pdf)
to build tis core library.

## Things that Are or Will be a Part of this Library
### (We Can do Easily in Pure Python)
- [x] Vectormath ([already exists in Ladybug core](https://github.com/ladybug-tools/ladybug/blob/master/ladybug/euclid.py))
- [x] Calculate Bounding Box ([already exists in Butterfly core](https://github.com/ladybug-tools/butterfly/blob/master/butterfly/geometry.py))
- [x] Compute Triangle + Quad Areas, Center Points + Normals ([partly exists in Butterfly core](https://github.com/ladybug-tools/butterfly/blob/master/butterfly/geometry.py))
- [x] Compute Area + Perimeter of Planar Geometry (should be doable [by using this formula](https://www.mathopenref.com/coordpolygonarea.html))
- [x] Check Concavity of a 2D Geometry (already exists in legacy [find non-convex component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_Honeybee.py#L9340-L9410))
- [x] Convert Concave 2D Geometry to Convex Geometries (should be possible with the [ear clipping method](https://en.wikipedia.org/wiki/Polygon_triangulation))
- [x] Triangulate Planar Geometry ([possible by converting convex geometry to concave and using fan triangulation](https://en.wikipedia.org/wiki/Polygon_triangulation))
- [x] Move Geometry (can be taken from Rhino3dm)
- [x] Rotate Geometry Around a Base Point (can be taken from Rhino3dm)
- [x] Mirror Geometry (can be taken from Rhino3dm)
- [x] Scale Geometry from a Base Point (can be taken from Rhino3dm)
- [x] Is Point Inside 2D Polygon (look pretty straightforward from [this example](https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/))
- [x] Planar surface grid generation ([as done in this thesis](https://www.radiance-online.org/community/workshops/2015-philadelphia/presentations/day1/STADICUtilities-Radiance%20Workshop2015.pdf), which uses bounding box and is point inside)
- [x] Glazing Based on Ratio (currently implemented in legacy [glazing based on ratio component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_Glazing%20based%20on%20ratio.py))
- [x] Solve Adjacencies (possible with good geometric equivalency tests)
- [x] Generate louvers, fins and overhangs from a face. Should be easy by extruding intersected line segments generated from the Face3D.intersect_plane method that is already implemented.
- [x] Check if a 3D PolyFace is Closed ([should be possible by creating a 3D triangulated mesh](https://gamedev.stackexchange.com/questions/61878/how-check-if-an-arbitrary-given-mesh-is-a-single-closed-mesh/61886))
- [x] Ensure that all faces of a solid PolyFace are facing outward (possible with a 3D version of the current method that checks whether a point is inside a polygon. Essentially, shoot a ray from the center of a face using the face normal and make sure that the ray intersects 0 or an even number of other faces in the polyface)
- [x] Check if a Point is Inside a Closed 3D Geometry (useful for thermal comfort studies where points in a grid must be matched with zone results) (should be possible with a variation of the function that checks if a face is outward-facing except we use the point in question as the origin of the ray we shoot)
- [ ] Straight Skeleton Methods (currently implemented in [legacy core/perimeter component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_SplitFloor2ThermalZones.py) but should be expanded to accept concave geometry)
- [ ] Offset edge curve of a planar surface (can be done by translating vertices along the straight skeleton to make a "wavefront")

## Things That Should be a Part of this Library
### (We Think We Can Do Them But They Will Require Some Expertise)
- [ ] Create Matching Zone Surfaces (intersection of surfaces with one another). OpenStudio has methods for this [as @saeranv shows here](https://github.com/mostaphaRoudsari/honeybee/issues/700)
- [ ] Curve Boolean a set of 2D curves (useful for finding outer boundaries of set of THERM polygons, calculating building footprints from floor curves, and more).  Should be possible with [this method here](https://stackoverflow.com/questions/2667748/how-do-i-combine-complex-polygons)

## Things that We Will Rely on the CAD Interface For:
- Conversion of Curved Surfaces to Planar Surfaces (ideally with methods for treating single curvature differently than double curvature)
- Solid Boolean Unions (we can probably get away with not needing this for anything in Ladybug Tools)
