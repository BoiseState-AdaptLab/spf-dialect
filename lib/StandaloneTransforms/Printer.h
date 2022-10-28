#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h" // imports something we need for some reason IDK
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir;

struct NewLineCounter {
  unsigned curLine = 1;
};

/// CopyPastedPrintingStuff is a lightly modified copy pasted version of
/// AsmPrinter::Impl in llvm-project/mlir/lib/IR/AsmPrinter.cpp at llvm-project
/// sha: 657bfa3. The modifications change the way maps are printed to conform
/// to the string representation IEGenLib Relation parser expects.
///
/// It would be nice if they provided a layered API
/// (https://caseymuratori.com/blog_0024). A layered API provides access to base
/// level things suitable for low level access/control, then complex wrappers
/// around them ergonomic for specific needs. For example, if they had
/// printAffineMap and printAffineExpression as simple functions available
/// publicly. Then, they could have the private AsmPrinter::Impl use those
/// functions in the specific way that is convenient for the stuff they are
/// doing.  Alas, there doesn't seem to be a way to get at this stuff except
/// through the complex wrappers which are designed for problems and situations
/// that I don't have currently. So copy paste this crap it is!
class CopyPastedPrintingStuff {
public:
  CopyPastedPrintingStuff(llvm::raw_ostream &os) : os(os) {}

  template <typename Container, typename UnaryFunctor>
  inline void interleaveComma(const Container &c, UnaryFunctor eachFn) const {
    llvm::interleaveComma(c, os, eachFn);
  }

  void printAffineMap(AffineMap map);

  void
  printAffineExpr(AffineExpr expr,
                  function_ref<void(unsigned, bool)> printValueName = nullptr);

  enum class BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
  };

  void printAffineExprInternal(
      AffineExpr expr, BindingStrength enclosingTightness,
      function_ref<void(unsigned, bool)> printValueName = nullptr);

  /// The output stream for the printer.
  llvm::raw_ostream &os;
};

inline void CopyPastedPrintingStuff::printAffineMap(AffineMap map) {
  // open relation
  os << '{';

  // Dimension identifiers.
  os << '[';
  for (int i = 0; i < (int)map.getNumDims() - 1; ++i)
    os << 'd' << i << ", ";
  if (map.getNumDims() >= 1)
    os << 'd' << map.getNumDims() - 1;
  os << ']';

  // Symbolic identifiers.
  if (map.getNumSymbols() != 0) {
    os << '[';
    for (unsigned i = 0; i < map.getNumSymbols() - 1; ++i)
      os << 's' << i << ", ";
    if (map.getNumSymbols() >= 1)
      os << 's' << map.getNumSymbols() - 1;
    os << ']';
  }

  // Result affine expressions.
  os << " -> [";
  interleaveComma(map.getResults(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ']';

  // close relation
  os << '}';
}

inline void CopyPastedPrintingStuff::printAffineExpr(
    AffineExpr expr, function_ref<void(unsigned, bool)> printValueName) {
  printAffineExprInternal(expr, BindingStrength::Weak, printValueName);
}

inline void CopyPastedPrintingStuff::printAffineExprInternal(
    AffineExpr expr, BindingStrength enclosingTightness,
    function_ref<void(unsigned, bool)> printValueName) {
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId: {
    unsigned pos = expr.cast<AffineSymbolExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/true);
    else
      os << 's' << pos;
    return;
  }
  case AffineExprKind::DimId: {
    unsigned pos = expr.cast<AffineDimExpr>().getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/false);
    else
      os << 'd' << pos;
    return;
  }
  case AffineExprKind::Constant:
    os << expr.cast<AffineConstantExpr>().getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  auto binOp = expr.cast<AffineBinaryOpExpr>();
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    // Pretty print multiplication with -1.
    auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>();
    if (rhsConst && binOp.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << "-";
      printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }

    printAffineExprInternal(lhsExpr, BindingStrength::Strong, printValueName);

    os << binopSpelling;
    printAffineExprInternal(rhsExpr, BindingStrength::Strong, printValueName);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = rhsExpr.dyn_cast<AffineBinaryOpExpr>()) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = rrhsExpr.dyn_cast<AffineConstantExpr>()) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                    printValueName);
          } else {
            printAffineExprInternal(rhs.getLHS(), BindingStrength::Weak,
                                    printValueName);
          }

          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExprInternal(lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          printAffineExprInternal(rhs.getLHS(), BindingStrength::Strong,
                                  printValueName);
          os << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhsConst = rhsExpr.dyn_cast<AffineConstantExpr>()) {
    if (rhsConst.getValue() < 0) {
      printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);
      os << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(lhsExpr, BindingStrength::Weak, printValueName);

  os << " + ";
  printAffineExprInternal(rhsExpr, BindingStrength::Weak, printValueName);

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
}