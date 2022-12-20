#include "Utils.h"
#include "mlir/IR/MLIRContext.h"
#include <tuple>

// TODO: copying all these strings around is definitely bad.
std::string unmangleUfName(std::string ufMangledName) {
  // This is all a hack, it seems like the omega rename thing in hte
  // computation API ads "_<some number>", so I'm just removing that
  // here. There's definitely some way to compose the problem so
  // that this isn't a problem, and that way would be better.
  auto underscorePos = ufMangledName.find("_");
  return ufMangledName.substr(0, underscorePos);
}