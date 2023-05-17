
#include "../datastructures.hpp"
#include "../projHelper.hpp"
#include <cstddef>
#include <memory>

namespace roofer {
  struct RasterWriterInterface {

    projHelperInterface& pjHelper;

    RasterWriterInterface(projHelperInterface& pjh) : pjHelper(pjh) {};

    virtual void writeBands(
      const std::string& source, 
      ImageMap& bands) = 0;
  };

  std::unique_ptr<RasterWriterInterface> createRasterWriterGDAL(projHelperInterface& pjh);
}