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
// Ravi Peters
// Balazs Dukai

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <roofer/io/CityJsonWriter.hpp>
#include <set>
#include <sstream>

namespace roofer::io {

  namespace fs = std::filesystem;

  class CityJsonWriter : public CityJsonWriterInterface {
    template <typename T>
    void add_vertices_ring(std::map<arr3d, size_t>& vertex_map,
                           std::vector<arr3d>& vertex_vec,
                           std::set<arr3d>& vertex_set, const T& ring,
                           TBox<double>& bbox) {
      size_t v_cntr = vertex_vec.size();
      for (auto& vertex_ : ring) {
        auto vertex = pjHelper.coord_transform_rev(vertex_);
        bbox.add(vertex);
        auto [it, did_insert] = vertex_set.insert(vertex);
        if (did_insert) {
          vertex_map[vertex] = v_cntr++;
          vertex_vec.push_back(vertex);
        }
      }
    }

    TBox<double> add_vertices_polygon(std::map<arr3d, size_t>& vertex_map,
                                      std::vector<arr3d>& vertex_vec,
                                      std::set<arr3d>& vertex_set,
                                      const LinearRing& polygon) {
      TBox<double> bbox;
      add_vertices_ring(vertex_map, vertex_vec, vertex_set, polygon, bbox);
      for (auto& iring : polygon.interior_rings()) {
        add_vertices_ring(vertex_map, vertex_vec, vertex_set, iring, bbox);
      }
      return bbox;
    }

    TBox<double> add_vertices_mesh(std::map<arr3d, size_t>& vertex_map,
                                   std::vector<arr3d>& vertex_vec,
                                   std::set<arr3d>& vertex_set,
                                   const Mesh& mesh) {
      TBox<double> bbox;
      for (auto& face : mesh.get_polygons()) {
        bbox.add(
            add_vertices_polygon(vertex_map, vertex_vec, vertex_set, face));
      }
      return bbox;
    }

    std::vector<std::vector<size_t>> LinearRing2jboundary(
        std::map<arr3d, size_t>& vertex_map, const LinearRing& face) {
      std::vector<std::vector<size_t>> jface;
      std::vector<size_t> exterior_ring;
      for (auto& vertex_ : face) {
        auto vertex = pjHelper.coord_transform_rev(vertex_);
        exterior_ring.push_back(vertex_map[vertex]);
      }
      jface.emplace_back(std::move(exterior_ring));
      for (auto& iring : face.interior_rings()) {
        std::vector<size_t> interior_ring;
        for (auto& vertex_ : iring) {
          auto vertex = pjHelper.coord_transform_rev(vertex_);
          interior_ring.push_back(vertex_map[vertex]);
        }
        jface.emplace_back(std::move(interior_ring));
      }
      return jface;
    }

    nlohmann::json::object_t mesh2jSolid(const Mesh& mesh, const char* lod,
                                         std::map<arr3d, size_t>& vertex_map) {
      auto geometry = nlohmann::json::object();
      geometry["type"] = "Solid";
      geometry["lod"] = lod;
      std::vector<std::vector<std::vector<size_t>>> exterior_shell;

      for (auto& face : mesh.get_polygons()) {
        exterior_shell.emplace_back(LinearRing2jboundary(vertex_map, face));
      }
      geometry["boundaries"] = {exterior_shell};

      auto semantic_objects = nlohmann::json::array();
      // 0
      semantic_objects.push_back(
          nlohmann::json::object({{"type", "GroundSurface"}}));
      // 1
      semantic_objects.push_back(nlohmann::json::object(
          {{"type", "WallSurface"}, {"on_footprint_edge", true}}));
      // 2
      semantic_objects.push_back(nlohmann::json::object(
          {{"type", "WallSurface"}, {"on_footprint_edge", false}}));

      std::vector<int> sem_values;
      size_t wallSurface_cntr = semantic_objects.size();
      for (size_t i = 0; i < mesh.get_polygons().size(); ++i) {
        auto& label = mesh.get_labels()[i];
        if (label == 0) {  // GroundSurface
          sem_values.push_back(0);
        } else if (label == 1) {  // RoofSurface
          nlohmann::json::object_t semantic_object;
          if (mesh.get_attributes().size()) {
            semantic_object = attributes2json(mesh.get_attributes().at(i));
          }
          semantic_object["type"] = "RoofSurface";
          semantic_objects.push_back(semantic_object);
          sem_values.push_back(wallSurface_cntr++);
        } else if (label == 2) {  // WallSurface on footprint edge
          sem_values.push_back(1);
        } else if (label == 3) {  // WallSurface not on footprint edge
          sem_values.push_back(2);
        } else {
          throw rooferException("Unknown label in mesh");
        }
      }
      geometry["semantics"] = {{"surfaces", semantic_objects},
                               {"values", {sem_values}}};
      return geometry;
    }

    void write_to_stream(const nlohmann::json& outputJSON,
                         std::ostream& output_stream, bool prettyPrint_) {
      output_stream << std::fixed << std::setprecision(2);
      try {
        if (prettyPrint_)
          output_stream << outputJSON.dump(2);
        else
          output_stream << outputJSON;

        output_stream << std::endl;
      } catch (const std::exception& e) {
        throw(rooferException(e.what()));
      }
    }

    // Computes the geographicalExtent array from a geoflow::Box and the
    // data_offset from the NodeManager
    nlohmann::json::array_t compute_geographical_extent(
        const TBox<double>& bbox) {
      auto minp = bbox.min();
      auto maxp = bbox.max();
      return {minp[0], minp[1], minp[2], maxp[0], maxp[1], maxp[2]};
    }

    nlohmann::json::object_t attributes2json(
        const AttributeMapRow& attributes) {
      nlohmann::json::object_t jattributes;
      nlohmann::json j_null;
      for (const auto& [name, val] : attributes) {
        if (attributes.is_null(name)) {
          jattributes[name] = j_null;
        } else if (auto val = attributes.get_if<bool>(name)) {
          jattributes[name] = *val;
        } else if (auto val = attributes.get_if<float>(name)) {
          jattributes[name] = *val;
        } else if (auto val = attributes.get_if<int>(name)) {
          jattributes[name] = *val;
        } else if (auto val = attributes.get_if<std::string>(name)) {
          jattributes[name] = *val;
          // for date/time we follow https://en.wikipedia.org/wiki/ISO_8601
        } else if (auto val = attributes.get_if<Date>(name)) {
          auto t = *val;
          std::string date = t.format_to_ietf();
          jattributes[name] = date;
        } else if (auto val = attributes.get_if<Time>(name)) {
          auto t = *val;
          std::string time = std::to_string(t.hour) + ":" +
                             std::to_string(t.minute) + ":" +
                             std::to_string(t.second) + "Z";
          jattributes[name] = time;
        } else if (auto val = attributes.get_if<DateTime>(name)) {
          auto t = *val;
          std::string datetime = t.format_to_ietf();
          jattributes[name] = datetime;
        }
      }
      return jattributes;
    }

    using MeshMap = std::unordered_map<int, Mesh>;
    void write_cityobject(
        const LinearRing& footprint, const MeshMap* multisolid_lod12,
        const MeshMap* multisolid_lod13, const MeshMap* multisolid_lod22,
        const AttributeMapRow& attributes, nlohmann::json& outputJSON,
        std::vector<arr3d>& vertex_vec, std::string& identifier_attribute,
        std::string building_id) {
      std::map<arr3d, size_t> vertex_map;
      std::set<arr3d> vertex_set;
      size_t bp_counter = 0;

      // we expect at least one of the geomtry inputs is set
      bool export_lod12 = multisolid_lod12;
      bool export_lod13 = multisolid_lod13;
      bool export_lod22 = multisolid_lod22;

      {
        auto building = nlohmann::json::object();
        auto b_id = building_id;
        building["type"] = "Building";

        // Building atributes
        // bool id_from_attr = false;
        building["attributes"] = attributes2json(attributes);
        for (const auto& [name, val] : attributes) {
          if (name == identifier_attribute) {
            if (auto val = attributes.get_if<float>(name)) {
              b_id = std::to_string(*val);
            } else if (auto val = attributes.get_if<int>(name)) {
              b_id = std::to_string(*val);
            } else if (auto val = attributes.get_if<std::string>(name)) {
              b_id = *val;
            }
          }
        }

        // footprint geometry
        auto fp_geometry = nlohmann::json::object();
        fp_geometry["lod"] = "0";
        fp_geometry["type"] = "MultiSurface";

        add_vertices_polygon(vertex_map, vertex_vec, vertex_set, footprint);
        fp_geometry["boundaries"] = {
            LinearRing2jboundary(vertex_map, footprint)};
        building["geometry"].push_back(fp_geometry);

        std::vector<std::string> buildingPartIds;

        bool has_solids = false;
        if (export_lod12) has_solids = multisolid_lod12->size();
        if (export_lod13) has_solids = multisolid_lod13->size();
        if (export_lod22) has_solids = multisolid_lod22->size();

        TBox<double> building_bbox;
        if (has_solids) {
          const MeshMap* meshmap;
          if (export_lod22) {
            meshmap = multisolid_lod22;
          } else if (export_lod13) {
            meshmap = multisolid_lod13;
          } else if (export_lod12) {
            meshmap = multisolid_lod12;
          }
          for (const auto& [sid, solid_lodx] : *meshmap) {
            auto buildingPart = nlohmann::json::object();
            auto bp_id = b_id + "-" + std::to_string(sid);

            buildingPartIds.push_back(bp_id);
            buildingPart["type"] = "BuildingPart";
            buildingPart["parents"] = {b_id};

            // Use try-except here for some rare cases when the sid's between
            // different lod's do not line up (eg for very fragmented buildings
            // from poor dim pointcloud).
            if (export_lod12) {
              try {
                building_bbox =
                    add_vertices_mesh(vertex_map, vertex_vec, vertex_set,
                                      multisolid_lod12->at(sid));
                buildingPart["geometry"].push_back(
                    mesh2jSolid(multisolid_lod12->at(sid), "1.2", vertex_map));
              } catch (const std::exception& e) {
                // std::cout << "skipping lod 12 building part\n";
              }
            }
            if (export_lod13) {
              try {
                building_bbox =
                    add_vertices_mesh(vertex_map, vertex_vec, vertex_set,
                                      multisolid_lod13->at(sid));
                buildingPart["geometry"].push_back(
                    mesh2jSolid(multisolid_lod13->at(sid), "1.3", vertex_map));
              } catch (const std::exception& e) {
                // std::cout << "skipping lod 13 building part\n";
              }
            }
            if (export_lod22) {
              building_bbox =
                  add_vertices_mesh(vertex_map, vertex_vec, vertex_set,
                                    multisolid_lod22->at(sid));
              buildingPart["geometry"].push_back(
                  mesh2jSolid(multisolid_lod22->at(sid), "2.2", vertex_map));
            }

            // attributes
            //  auto jattributes = nlohmann::json::object();
            //  for (auto& term : part_attributes.sub_terminals()) {
            //    auto tname = term->get_full_name();
            //    if (!term->get_data_vec()[i].has_value()) {
            //      nlohmann::json j_null;
            //      jattributes[tname] = j_null;
            //      continue;
            //    }

            //   if (term->accepts_type(typeid(bool))) {
            //     jattributes[tname] = term->get<const bool&>(bp_counter);
            //   } else if (term->accepts_type(typeid(float))) {
            //     jattributes[tname] = term->get<const float&>(bp_counter);
            //   } else if (term->accepts_type(typeid(int))) {
            //     jattributes[tname] = term->get<const int&>(bp_counter);
            //   } else if (term->accepts_type(typeid(std::string))) {
            //     jattributes[tname] = term->get<const
            //     std::string&>(bp_counter);
            //   }
            // }
            // ++bp_counter;
            // buildingPart["attributes"] = jattributes;

            outputJSON["CityObjects"][bp_id] = buildingPart;
          }
        }

        building["children"] = buildingPartIds;
        building["geographicalExtent"] =
            compute_geographical_extent(building_bbox);

        outputJSON["CityObjects"][b_id] = building;
      }
    }

   public:
    using CityJsonWriterInterface::CityJsonWriterInterface;

    void write_metadata(std::ostream& output_stream,
                        const SpatialReferenceSystemInterface* srs,
                        const roofer::TBox<double>& extent,
                        CityJSONMetadataProperties props) override {
      // metadata
      nlohmann::json outputJSON;
      outputJSON["type"] = "CityJSON";
      outputJSON["version"] = "2.0";
      outputJSON["CityObjects"] = nlohmann::json::object();
      outputJSON["vertices"] = nlohmann::json::array();

      outputJSON["transform"] = {
          {"scale", {scale_x_, scale_y_, scale_z_}},
          {"translate", {translate_x_, translate_y_, translate_z_}}};

      auto metadata = nlohmann::json::object();
      metadata["geographicalExtent"] = compute_geographical_extent(extent);

      if (props.identifier.size()) metadata["identifier"] = props.identifier;

      // metadata.datasetPointOfContact - only add it if at least one of the
      // parameters is filled
      auto contact = nlohmann::json::object();

      if (props.poc_contactName.size()) {
        contact["contactName"] = props.poc_contactName;
      }
      if (props.poc_emailAddress.size()) {
        contact["emailAddress"] = props.poc_emailAddress;
      }
      if (props.poc_phone.size()) {
        contact["phone"] = props.poc_phone;
      }
      // if (props.poc_address.size()) { contact["address"] = props.poc_address;
      // }
      if (props.poc_contactType.size()) {
        contact["contactType"] = props.poc_contactType;
      }
      if (props.poc_website.size()) {
        contact["website"] = props.poc_website;
      }

      if (contact.size()) metadata["pointOfContact"] = contact;

      if (props.referenceDate.empty()) {
        // find current date if none provided
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d");
        props.referenceDate = oss.str();
      }
      metadata["referenceDate"] = props.referenceDate;

      // metadata["referenceSystem"] =
      // manager.substitute_globals(meta_referenceSystem_);
      if (srs->is_valid()) {
        metadata["referenceSystem"] = "https://www.opengis.net/def/crs/" +
                                      srs->get_auth_name() + "/0/" +
                                      srs->get_auth_code();
      }

      if (props.title.size()) metadata["title"] = props.title;

      if (metadata.size()) outputJSON["metadata"] = metadata;

      write_to_stream(outputJSON, output_stream, prettyPrint_);
    }

    void write_feature(std::ostream& output_stream, const LinearRing& footprint,
                       const MeshMap* multisolid_lod12,
                       const MeshMap* multisolid_lod13,
                       const MeshMap* multisolid_lod22,
                       const AttributeMapRow attributes) override {
      nlohmann::json outputJSON;

      outputJSON["type"] = "CityJSONFeature";
      outputJSON["CityObjects"] = nlohmann::json::object();

      std::vector<arr3d> vertex_vec;
      write_cityobject(footprint, multisolid_lod12, multisolid_lod13,
                       multisolid_lod22, attributes, outputJSON, vertex_vec,
                       identifier_attribute,
                       std::to_string(++written_features_count));

      // The main Building is the parent object.
      // Bit of a hack. Ideally we would know exactly which ID we set,
      // instead of just iterating. But it is assumed that in case of writing to
      // CityJSONFeature there is only one parent CityObject.
      for (auto& el : outputJSON["CityObjects"].items()) {
        if (!el.value().contains(std::string("parents"))) {
          outputJSON["id"] = el.key();
        }
      };

      std::vector<std::array<int, 3>> vertices_int;
      for (auto& vertex : vertex_vec) {
        vertices_int.push_back({int((vertex[0] - translate_x_) / scale_x_),
                                int((vertex[1] - translate_y_) / scale_y_),
                                int((vertex[2] - translate_z_) / scale_z_)});
      }
      outputJSON["vertices"] = vertices_int;

      write_to_stream(outputJSON, output_stream, prettyPrint_);
    }
  };

  std::unique_ptr<CityJsonWriterInterface> createCityJsonWriter(
      roofer::misc::projHelperInterface& pjh) {
    return std::make_unique<CityJsonWriter>(pjh);
  };
}  // namespace roofer::io
