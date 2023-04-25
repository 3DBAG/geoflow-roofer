
#include "datastructures.hpp"
#include "projHelper.hpp"
#include <memory>

namespace roofer {
  struct VectorReaderInterface {

    projHelperInterface& pjHelper;

    VectorReaderInterface(projHelperInterface& pjh) : pjHelper(pjh) {};

    virtual void open(const std::string& source) = 0;

    virtual std::vector<LinearRing> readPolygons() = 0;
  };

  std::unique_ptr<VectorReaderInterface> createVectorReaderOGR(projHelperInterface& pjh);
}