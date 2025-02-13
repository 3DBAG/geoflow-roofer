// Copyright (c) 2018-2024 TU Delft 3D geoinformation group, Ravi Peters (3DGI),
// and Balazs Dukai (3DGI)

// This file is part of roofer (https://github.com/3DBAG/roofer)

// geoflow-roofer was created as part of the 3DBAG project by the TU Delft 3D
// geoinformation group (3d.bk.tudelf.nl) and 3DGI (3dgi.nl)

// geoflow-roofer is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. geoflow-roofer is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details. You should have received a copy of the GNU
// General Public License along with geoflow-roofer. If not, see
// <https://www.gnu.org/licenses/>.

// Author(s):
// Ivan Paden

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <roofer/roofer.h>

namespace py = pybind11;

typedef std::vector<std::array<float, 3>> PyPointCollection;
typedef std::vector<std::vector<std::array<float, 3>>> PyLinearRing;
typedef std::vector<PyLinearRing> PyMesh;
typedef std::vector<std::array<size_t, 3>> PyFaceCollection;

namespace roofer {
  void convert_to_linear_ring(const PyLinearRing& footprint,
                              roofer::LinearRing& linear_ring) {
    for (const auto& p : footprint.front()) {
      float x = p[0];
      float y = p[1];
      float z = p[2];
      linear_ring.push_back({x, y, z});
    }
    for (size_t i = 1; i < footprint.size(); ++i) {
      roofer::LinearRing iring;
      for (const auto& p : footprint[i]) {
        float x = p[0];
        float y = p[1];
        float z = p[2];
        iring.push_back({x, y, z});
      }
      linear_ring.interior_rings().push_back(iring);
    }
  }

  std::vector<PyMesh> convert_meshes_to_py_meshes(
      const std::vector<Mesh>& meshes) {
    std::vector<PyMesh> py_meshes;
    for (const auto& roofer_mesh : meshes) {
      PyMesh py_mesh;
      for (const auto& roofer_poly : roofer_mesh.get_polygons()) {
        PyLinearRing py_poly;
        std::vector<std::array<float, 3>> py_ring;
        py_ring.reserve(roofer_poly.size());
        for (const auto& pt : roofer_poly) {
          py_ring.push_back({pt[0], pt[1], pt[2]});
        }
        py_poly.push_back(py_ring);
        for (const auto& roofer_interior_ring : roofer_poly.interior_rings()) {
          std::vector<std::array<float, 3>> py_interior_ring;
          py_interior_ring.reserve(roofer_interior_ring.size());
          for (const auto& pt : roofer_interior_ring) {
            py_interior_ring.push_back({pt[0], pt[1], pt[2]});
          }
          py_poly.push_back(py_interior_ring);
        }
        py_mesh.push_back(py_poly);
      }
      py_meshes.push_back(py_mesh);
    }
    return py_meshes;
  }

  Mesh convert_py_mesh_to_mesh(const PyMesh& py_mesh) {
    std::vector<Mesh> meshes;
    Mesh mesh;
    for (const auto& py_poly : py_mesh) {
      LinearRing poly;
      for (const auto& pt : py_poly.front()) {
        poly.push_back({pt[0], pt[1], pt[2]});
      }
      for (size_t i = 1; i < py_poly.size(); ++i) {
        LinearRing iring;
        for (const auto& pt : py_poly[i]) {
          iring.push_back({pt[0], pt[1], pt[2]});
        }
        poly.interior_rings().push_back(iring);
      }
      mesh.push_polygon(poly, 0);
    }
    return mesh;
  }

  std::vector<PyMesh> py_reconstruct(
      const PyPointCollection& points_roof,
      const PyPointCollection& points_ground, const PyLinearRing& footprint,
      ReconstructionConfig cfg = ReconstructionConfig()) {
    PointCollection points_roof_pc, points_ground_pc;
    for (const auto& pt : points_roof) {
      points_roof_pc.push_back({pt[0], pt[1], pt[2]});
    }
    for (const auto& pt : points_ground) {
      points_ground_pc.push_back({pt[0], pt[1], pt[2]});
    }
    roofer::LinearRing linear_ring;
    convert_to_linear_ring(footprint, linear_ring);
    auto meshes =
        reconstruct(points_roof_pc, points_ground_pc, linear_ring, cfg);
    return convert_meshes_to_py_meshes(meshes);
  }

  std::vector<PyMesh> py_reconstruct(
      const PyPointCollection& points_roof, const PyLinearRing& footprint,
      ReconstructionConfig cfg = ReconstructionConfig()) {
    PointCollection points_roof_pc;
    for (const auto& pt : points_roof) {
      points_roof_pc.push_back({pt[0], pt[1], pt[2]});
    }
    roofer::LinearRing linear_ring;
    convert_to_linear_ring(footprint, linear_ring);
    auto meshes = reconstruct(points_roof_pc, linear_ring, cfg);
    return convert_meshes_to_py_meshes(meshes);
  }

  // todo move vertex-face data struct to cpp api?
  std::tuple<PyPointCollection, PyFaceCollection> py_triangulate_mesh(
      const PyMesh& mesh) {
    Mesh roofer_mesh = convert_py_mesh_to_mesh(mesh);
    auto tri_mesh = triangulate_mesh(roofer_mesh);

    std::map<arr3f, size_t> vertex_map;
    PyPointCollection vertices;
    PyFaceCollection faces;
    for (const auto& triangle : tri_mesh) {
      for (const auto& vertex : triangle) {
        if (vertex_map.find(vertex) == vertex_map.end()) {
          vertex_map[vertex] = vertex_map.size();
          vertices.push_back(vertex);
        }
      }
      faces.push_back({vertex_map[triangle[0]], vertex_map[triangle[1]],
                       vertex_map[triangle[2]]});
    }
    return std::make_tuple(vertices, faces);
  }

}  // namespace roofer

PYBIND11_MODULE(roofer, m) {
  py::class_<roofer::ReconstructionConfig>(m, "ReconstructionConfig")
      .def(py::init<>())
      .def_readwrite("complexity_factor",
                     &roofer::ReconstructionConfig::complexity_factor)
      .def_readwrite("clip_ground", &roofer::ReconstructionConfig::clip_ground)
      .def_readwrite("lod", &roofer::ReconstructionConfig::lod)
      .def_readwrite("lod13_step_height",
                     &roofer::ReconstructionConfig::lod13_step_height)
      .def_readwrite("floor_elevation",
                     &roofer::ReconstructionConfig::floor_elevation)
      .def_readwrite(
          "override_with_floor_elevation",
          &roofer::ReconstructionConfig::override_with_floor_elevation)
      .def_readwrite("plane_detect_k",
                     &roofer::ReconstructionConfig::plane_detect_k)
      .def_readwrite("plane_detect_min_points",
                     &roofer::ReconstructionConfig::plane_detect_min_points)
      .def_readwrite("plane_detect_epsilon",
                     &roofer::ReconstructionConfig::plane_detect_epsilon)
      .def_readwrite("plane_detect_normal_angle",
                     &roofer::ReconstructionConfig::plane_detect_normal_angle)
      .def_readwrite("line_detect_epsilon",
                     &roofer::ReconstructionConfig::line_detect_epsilon)
      .def_readwrite("thres_alpha", &roofer::ReconstructionConfig::thres_alpha)
      .def_readwrite("thres_reg_line_dist",
                     &roofer::ReconstructionConfig::thres_reg_line_dist)
      .def_readwrite("thres_reg_line_ext",
                     &roofer::ReconstructionConfig::thres_reg_line_ext)
      .def("is_valid", &roofer::ReconstructionConfig::is_valid);

  m.def("reconstruct",
        py::overload_cast<const PyPointCollection&, const PyPointCollection&,
                          const PyLinearRing&, roofer::ReconstructionConfig>(
            &roofer::py_reconstruct),
        "Reconstruct a single instance of a building from a point cloud with "
        "ground points",
        py::arg("points_roof"), py::arg("points_ground"), py::arg("footprint"),
        py::arg("cfg") = roofer::ReconstructionConfig());

  m.def(
      "reconstruct",
      py::overload_cast<const PyPointCollection&, const PyLinearRing&,
                        roofer::ReconstructionConfig>(&roofer::py_reconstruct),
      "Reconstruct a single instance of a building from a point cloud "
      "without ground points",
      py::arg("points_roof"), py::arg("footprint"),
      py::arg("cfg") = roofer::ReconstructionConfig());

  m.def("triangulate_mesh", &roofer::py_triangulate_mesh, "Triangulate a mesh",
        py::arg("mesh"));
}
