# ladybug-geometry
A library to house all of the basic geometry computation needed for Ladybug Tools core libraries.

# Reasons for this Library
We had debated for a long time whether geometry computation should be placed largely on the plugins,
whether it should be in the core and, if so, how should it be done?  Realizing that there are large
advantages to having it in the core (ie. cross compatability, ability to process more from command line),
we decided to include geomtery computation as part of the core.

We looked into using other geometry computation libraries for the core including:
- [Rhino3dm](https://github.com/mcneel/rhino3dm)
- [Blender API (bpy)](https://docs.blender.org/api/current/)
However, Rhino3dm lacks basics types of computation that is needed in the core (like generating a
grid of points from a surface).
Furthermore, Blender library only works in Python3 and this would break our workflows for the
plugins for Grasshopper and Dynamo, which rely on IronPython.

After considering it further, we realized that many of the calculations that we need can be done
easily as long as the geomtry is planar.  Since all of the geomtry going to the engines (Radiance, E+)
is eventually converted to a planar format anyway, we made the decision that the core libraries will only support
geometry computation for planar objects.  With this, it should be possible to build out a good geomtry library
in pure python (this repository).  We can borrow some of the math from the previous open source libaraies 
listed above (Rhino3dm, Blender), and other projects like 
[this PhD on Grid Generation for Radiance](https://www.radiance-online.org/community/workshops/2015-philadelphia/presentations/day1/STADICUtilities-Radiance%20Workshop2015.pdf)
to build tis core library.

# Things that Will be a Part of this Library (we can do in pure python):
- [ ] Is Point Inside Polygon
- [ ] Planar surface grid generation
- [ ] Glazing Based on Ratio
- [ ] Vectormath
- [ ] Calculate Bounding Box
- [ ] Move Geometry
- [ ] Rotate Geometry Around a Base Point
- [ ] Mirror Geometry
- [ ] Scale Geometry from a Base Point
- [ ] Check Concavity (in 2D)
- [ ] Concave to Convex (in 2D)
- [ ] Straight Skeleton
- [ ] Triangulate planar geometry

# Things that may require more expertise but we should be able to do in Python:
- [ ] Check if a 3D Geometry is Closed
- [ ] Check if a Point is Inside Closed Geometry

# Things that we will rely on the Plugins to do:
- Conversion of Curved Surfaces to Planar Surfaces
- Extracting Points from Planar Geometry
- Split Mass to Floors (Intersection of a closed volume to a plane)

# This should integrate across:
- Butterfly Geometry (mesh and vector math)
- Ladybug Euclid

