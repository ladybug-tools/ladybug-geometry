import os
from ladybug_geometry.interop.stl import STL


def test_stl_ascii():
    """Test STL class with an ascii STL file."""
    file_path = 'tests/stl/cube_ascii.stl'
    stl_obj = STL.from_file(file_path)

    assert isinstance(stl_obj, STL)
    assert stl_obj.name == 'OBJECT'
    assert len(stl_obj.face_vertices) == 12
    assert len(stl_obj.face_normals) == 12

    new_folder, new_name = 'tests/stl/', 'cube_ascii_rebuilt.stl'
    new_file = stl_obj.to_file(new_folder, new_name)
    assert os.path.isfile(new_file)
    new_stl_obj = STL.from_file(new_file)
    assert isinstance(new_stl_obj, STL)
    assert new_stl_obj.name == 'OBJECT'
    assert len(new_stl_obj.face_vertices) == 12
    assert len(new_stl_obj.face_normals) == 12
    os.remove(new_file)


def test_stl_binary():
    """Test  STL class with a binary STL file."""
    file_path = 'tests/stl/cube_binary.stl'
    stl_obj = STL.from_file(file_path)

    assert isinstance(stl_obj, STL)
    assert stl_obj.name == 'Rhinoceros Binary STL ( Apr 12 2022 )'
    assert len(stl_obj.face_vertices) == 12
    assert len(stl_obj.face_normals) == 12
