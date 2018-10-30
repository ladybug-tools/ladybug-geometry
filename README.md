# ladybug-geometry
A library to house all of the basic geometry computation needed for Ladybug Tools core libraries.

# Reasons for this Library
We initially debated whether geometry computation should be placed largely on the plugins or
whether it should be inincluded in the core.  As we developed the core libraries out, it became clear
that there are large advantages to having it in the core (ie. cross compatability between
the CAD plugins, ability to process more inputs from command line, and shear speed
since many of the CAD libraries are made to address many more use cases than we have to).
So we have decided to include geomtery computation as part of the Ladybug Tools core.

We looked into using other geometry computation libraries for the core including:
- [Rhino3dm](https://github.com/mcneel/rhino3dm)
- [Blender API (bpy)](https://docs.blender.org/api/current/)

However, Rhino3dm lacks basic types of computation that is needed in the core (like generating a
grid of points from a surface).
Furthermore, Blender library only works in Python3 and this would break our workflows for the
plugins for Grasshopper and Dynamo, which rely on IronPython.

After considering it further, we realized that many of the calculations that we need can be done
easily as long as the geomtry is planar.  Since all of the geometry going to the engines (Radiance, E+)
is eventually converted to a planar format anyway, we made the decision that the core libraries will support
geometry computation for planar objects only.  With this, it should be possible to build out a good
geometry library in pure python.  Thus this repository was born!

For this library, we can borrow some of the math from the previous open source libaraies 
listed above (Rhino3dm and Blender), as well as other projects like 
[this PhD on Grid Generation for Radiance](https://www.radiance-online.org/community/workshops/2015-philadelphia/presentations/day1/STADICUtilities-Radiance%20Workshop2015.pdf)
to build tis core library.

# Things that Will be a Part of this Library (we can do in pure python):
- [ ] Vectormath (already exists in Ladybug core)
- [ ] Calculate Bounding Box (already exists in Butterfly core)
- [ ] Check Concavity of a 2D Geometry (already exists in legacy [Find non-convex component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_Find%20Non-Convex.py))
- [ ] Convert Concave 2D Geometry to a Series of Convex Geometries (should be possible with the [ear clipping method](https://en.wikipedia.org/wiki/Polygon_triangulation))
- [ ] Compute Triangle and Quad Areas (very basic math here)
- [ ] Triangulate Planar Geometry (possible by converting convex geometry to concave and using [fan triangulation](https://en.wikipedia.org/wiki/Polygon_triangulation))
- [ ] Compute Area of Planar Geometry (built by triangulating geometry and computing the area of each triangle)
- [ ] Check if a 3D Geometry is Closed (should be possible by [creating a triangulated mesh](https://gamedev.stackexchange.com/questions/61878/how-check-if-an-arbitrary-given-mesh-is-a-single-closed-mesh/61886))
- [ ] Move Geometry (can be taken from Rhino3dm)
- [ ] Rotate Geometry Around a Base Point (can be taken from Rhino3dm)
- [ ] Mirror Geometry (can be taken from Rhino3dm)
- [ ] Scale Geometry from a Base Point (can be taken from Rhino3dm)
- [ ] Is Point Inside 2D Polygon (look pretty straightforward from [this example](https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/))
- [ ] Planar surface grid generation (per [this thesis](https://www.radiance-online.org/community/workshops/2015-philadelphia/presentations/day1/STADICUtilities-Radiance%20Workshop2015.pdf), which uses bounding box and is point inside)
- [ ] Glazing Based on Ratio (currently implemented in legacy [glazing based on ratio component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_Glazing%20based%20on%20ratio.py))
- [ ] Straight Skeleton Methods (currently implemented in [legacy core/perimeter component](https://github.com/mostaphaRoudsari/honeybee/blob/master/src/Honeybee_SplitFloor2ThermalZones.py))
- [ ] Solve Adjacencies (I think that OpenStudio team has some code for this)

# Things that may require more expertise but we should be able to do in Python:
- [ ] Check if a Point is Inside a Closed 3D Geometry

# Things that we will rely on the Plugins to do:
- Conversion of Curved Surfaces to Planar Surfaces
- Extract Vertices from Planar Geometry
- Split Closed 3D Volume to Floors (Intersection of a closed volume and a plane)

