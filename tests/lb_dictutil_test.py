"""Tests for Ladybug Geometry dict->Object converter.

Note: Written in PyTest format.
"""

from ladybug_geometry.geometry2d import (Vector2D, Point2D, Ray2D, 
                            LineSegment2D, Arc2D, Polyline2D, Polygon2D, Mesh2D)
from ladybug_geometry.geometry3d import (Vector3D, Point3D, Ray3D,
                            LineSegment3D, Arc3D, Polyline3D, Polyface3D, Mesh3D,
                            Plane, Face3D, Sphere, Cone, Cylinder)
from lb_dictutil import lb_geom_dict_to_object

#--- Test 2D Geometry
def test_Point2D():
    obj1 = Point2D()
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Arc2D():
    pt1 = Point2D()
    obj1 = Arc2D(pt1, 2)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Vector2D():
    obj1 = Vector2D(0,1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Ray2D():
    pt1 = Point2D()
    v1 = Vector2D(0,1)
    obj1 = Ray2D(pt1,v1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_LineSegment2D():
    pt1 = Point2D()
    v1 = Vector2D(0,1)
    obj1 = LineSegment2D(pt1,v1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Polyline2D():
    pt1 = Point2D(0,1)
    pt2 = Point2D(1,2)
    pt3 = Point2D(2,3)
    obj1 = Polyline2D([pt1, pt2, pt3], False)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Polygon2D():
    pt1 = Point2D(0,1)
    pt2 = Point2D(1,1)
    pt3 = Point2D(1,0)
    pt4 = Point2D(0,1)
    obj1 = Polygon2D([pt1, pt2, pt3, pt4])
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Mesh2D():
    pt1 = Point2D(0,1)
    pt2 = Point2D(1,1)
    pt3 = Point2D(1,0)
    pt4 = Point2D(0,1)
    obj1 = Mesh2D(vertices=(pt1, pt2, pt3, pt4), faces=[(0,1,2)] )
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

#--- Test 3D Geometry
def test_Point3D():
    obj1 = Point3D()
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Arc3D():
    pl = Plane()
    obj1 = Arc3D(pl, 2)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Vector3D():
    obj1 = Vector3D(0,1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Ray3D():
    pt1 = Point3D()
    v1 = Vector3D(0,1)
    obj1 = Ray3D(pt1,v1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_LineSegment3D():
    pt1 = Point3D()
    v1 = Vector3D(0,1)
    obj1 = LineSegment3D(pt1,v1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Polyline3D():
    pt1 = Point3D(0,1)
    pt2 = Point3D(1,2)
    pt3 = Point3D(2,3)
    obj1 = Polyline3D([pt1, pt2, pt3], False)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Polyface3D():
    pt1 = Point3D(0,1)
    pt2 = Point3D(1,1)
    pt3 = Point3D(1,0)
    pt4 = Point3D(0,1)
    obj1 = Polyface3D(vertices=[pt1, pt2, pt3, pt4], face_indices=[ [[1]] ] )
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Mesh3D():
    pt1 = Point3D(0,1)
    pt2 = Point3D(1,1)
    pt3 = Point3D(1,0)
    pt4 = Point3D(0,1)
    obj1 = Mesh3D(vertices=(pt1, pt2, pt3, pt4), faces=[(0,1,2)] )
    
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Plane():
    obj1 = Plane()
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Face3D():
    pt1 = Point3D(0,1)
    pt2 = Point3D(1,2)
    pt3 = Point3D(2,3)
    obj1 = Face3D([pt1, pt2, pt3])
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Sphere():
    pt1 = Point3D()
    obj1 = Sphere(pt1, 1)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Cone():
    v1 = Point3D()
    axis = Vector3D(0,0,1)
    obj1 = Cone(v1, axis, 45)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

def test_Cylinder():
    v1 = Point3D()
    axis = Vector3D(0,0,1)
    obj1 = Cylinder(v1, axis, 10)
    d = obj1.to_dict()
    obj2 = lb_geom_dict_to_object(d)

    assert obj2 == obj1

# if __name__ == '__main__':
#     test_Point2D()
#     test_Arc2D()
#     test_Vector2D()
#     test_Ray2D()
#     test_LineSegment2D()
#     test_Polyline2D()
#     test_Polygon2D()
#     test_Mesh2D()

#     test_Point3D()
#     test_Arc3D()
#     test_Vector3D()
#     test_Ray3D()
#     test_LineSegment3D()
#     test_Polyline3D()
#     test_Polyface3D()
#     test_Mesh3D()
#     test_Plane()
#     test_Face3D()
#     test_Sphere()
#     test_Cone()
#     test_Cylinder()