import os

from ladybug_geometry.geometry3d.pointvector import Point3D
from ladybug_geometry.geometry3d.mesh import Mesh3D
from ladybug_geometry.interop.obj import OBJ


def test_obj_with_material():
    """Test OBJ class with a material structure."""
    file_path = 'tests/obj/two_material_cubes.obj'
    transl_obj = OBJ.from_file(file_path)

    assert isinstance(transl_obj, OBJ)
    assert len(transl_obj.vertices) == 48
    assert len(transl_obj.faces) == 16
    assert len(transl_obj.vertex_texture_map) == 48

    assert len(transl_obj.material_structure) == 2
    assert transl_obj.material_structure[0] == ('diffuse_0', 0)
    assert transl_obj.material_structure[1] == ('Gem_Material', 8)

    file_path = 'tests/obj/two_material_cubes_edit.obj'
    transl_obj = OBJ.from_file(file_path)

    assert isinstance(transl_obj, OBJ)
    assert len(transl_obj.vertices) == 48
    assert len(transl_obj.faces) == 16
    assert len(transl_obj.vertex_texture_map) == 48

    assert len(transl_obj.material_structure) == 2
    assert transl_obj.material_structure[0] == ('diffuse_0', 0)
    assert transl_obj.material_structure[1] == ('Gem_Material', 8)

    new_folder, new_name = 'tests/obj/', 'two_material_cubes_new.obj'
    new_file = transl_obj.to_file(new_folder, new_name)
    assert os.path.isfile(new_file)
    with open(file_path, 'r') as file1:
        with open(new_file, 'r') as file2:
            file1.readline()
            file2.readline()
            for lin1, lin2 in zip(file1, file2):
                assert len(lin1.split()) == len(lin2.split())
    os.remove(new_file)


def test_obj_without_material():
    """Test OBJ class without a material structure."""
    file_path = 'tests/obj/sphere_no_material.obj'
    transl_obj = OBJ.from_file(file_path)

    assert isinstance(transl_obj, OBJ)
    assert len(transl_obj.vertices) == 153
    assert len(transl_obj.vertex_texture_map) == 153
    assert len(transl_obj.vertex_normals) == 153
    assert len(transl_obj.faces) == 128

    assert transl_obj.material_structure is None
    new_folder, new_name = 'tests/obj/', 'two_material_cubes_new.obj'
    new_file = transl_obj.to_file(new_folder, new_name)
    assert os.path.isfile(new_file)
    os.remove(new_file)


def test_meshes_to_obj_with_material():
    """Test the creation of OBJ with a material structure."""
    pts1 = (Point3D(0, 0, 2), Point3D(0, 2, 2), Point3D(2, 2, 2), Point3D(2, 0, 2))
    pts2 = (Point3D(2, 2, 2), Point3D(2, 4, 2), Point3D(4, 4, 2), Point3D(4, 2, 2))
    mesh1 = Mesh3D(pts1, [(0, 1, 2, 3)])
    mesh2 = Mesh3D(pts2, [(0, 1, 2, 3)])

    transl_obj = OBJ.from_mesh3ds((mesh1, mesh2), ('material_1', 'material_2'))
    assert len(transl_obj.vertices) == 8
    assert len(transl_obj.faces) == 2

    assert len(transl_obj.material_structure) == 2
    assert transl_obj.material_structure[0] == ('material_1', 0)
    assert transl_obj.material_structure[1] == ('material_2', 1)

    new_folder, new_name = 'tests/obj/', 'two_faces.obj'
    new_file = transl_obj.to_file(new_folder, new_name, include_mtl=True)
    assert os.path.isfile(new_file)
    assert os.path.isfile(new_file.replace('.obj', '.mtl'))
    os.remove(new_file)
    os.remove(new_file.replace('.obj', '.mtl'))
