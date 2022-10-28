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

class CopyPastedPrintingStuff {
public:
  CopyPastedPrintingStuff(llvm::raw_ostream &os) : os(os) {}

  //   /// Returns the output stream of the printer.
  //   std::ostream &getStream() { return os; }

  template <typename Container, typename UnaryFunctor>
  inline void interleaveComma(const Container &c, UnaryFunctor eachFn) const {
    llvm::interleaveComma(c, os, eachFn);
  }

  //   /// This enum describes the different kinds of elision for the type of an
  //   /// attribute when printing it.
  //   enum class AttrTypeElision {
  //     /// The type must not be elided,
  //     Never,
  //     /// The type may be elided when it matches the default used in the
  //     parser
  //     /// (for example i64 is the default for integer attributes).
  //     May,
  //     /// The type must be elided.
  //     Must
  //   };

  //   /// Print the given attribute.
  //   void printAttribute(Attribute attr,
  //                       AttrTypeElision typeElision =
  //                       AttrTypeElision::Never);

  //   /// Print the alias for the given attribute, return failure if no alias
  //   could
  //   /// be printed.
  //   LogicalResult printAlias(Attribute attr);

  //   void printType(Type type);

  //   /// Print the alias for the given type, return failure if no alias could
  //   /// be printed.
  //   LogicalResult printAlias(Type type);

  //   /// Print the given location to the stream. If `allowAlias` is true, this
  //   /// allows for the internal location to use an attribute alias.
  //   void printLocation(LocationAttr loc, bool allowAlias = false);

  //   /// Print a reference to the given resource that is owned by the given
  //   /// dialect.
  //   void printResourceHandle(const AsmDialectResourceHandle &resource) {
  //     auto *interface = cast<OpAsmDialectInterface>(resource.getDialect());
  //     os << interface->getResourceKey(resource);
  //     dialectResources[resource.getDialect()].insert(resource);
  //   }

  void printAffineMap(AffineMap map);
  void
  printAffineExpr(AffineExpr expr,
                  function_ref<void(unsigned, bool)> printValueName = nullptr);
  //   void printAffineConstraint(AffineExpr expr, bool isEq);
  //   void printIntegerSet(IntegerSet set);

  // protected:
  //   void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
  //                              ArrayRef<StringRef> elidedAttrs = {},
  //                              bool withKeyword = false);
  //   void printNamedAttribute(NamedAttribute attr);
  //   void printTrailingLocation(Location loc, bool allowAlias = true);
  //   void printLocationInternal(LocationAttr loc, bool pretty = false);

  //   /// Print a dense elements attribute. If 'allowHex' is true, a hex string
  //   is
  //   /// used instead of individual elements when the elements attr is large.
  //   void printDenseElementsAttr(DenseElementsAttr attr, bool allowHex);

  //   /// Print a dense string elements attribute.
  //   void printDenseStringElementsAttr(DenseStringElementsAttr attr);

  //   /// Print a dense elements attribute. If 'allowHex' is true, a hex string
  //   is
  //   /// used instead of individual elements when the elements attr is large.
  //   void printDenseIntOrFPElementsAttr(DenseIntOrFPElementsAttr attr,
  //                                      bool allowHex);

  //   void printDialectAttribute(Attribute attr);
  //   void printDialectType(Type type);

  //   /// Print an escaped string, wrapped with "".
  //   void printEscapedString(StringRef str);

  //   /// Print a hex string, wrapped with "".
  //   void printHexString(StringRef str);
  //   void printHexString(ArrayRef<char> data);

  /// This enum is used to represent the binding strength of the enclosing
  /// context that an AffineExprStorage is being printed in, so we can
  /// intelligently produce parens.
  enum class BindingStrength {
    Weak,   // + and -
    Strong, // All other binary operators.
  };
  void printAffineExprInternal(
      AffineExpr expr, BindingStrength enclosingTightness,
      function_ref<void(unsigned, bool)> printValueName = nullptr);

  /// The output stream for the printer.
  llvm::raw_ostream &os;

  //   /// A set of flags to control the printer's behavior.
  //   OpPrintingFlags printerFlags;

  //   /// A tracker for the number of new lines emitted during printing.
  //   NewLineCounter newLine;

  //   /// A set of dialect resources that were referenced during printing.
  //   DenseMap<Dialect *, SetVector<AsmDialectResourceHandle>>
  //   dialectResources;
};

void CopyPastedPrintingStuff::printAffineMap(AffineMap map) {
  // Dimension identifiers.
  os << '(';
  for (int i = 0; i < (int)map.getNumDims() - 1; ++i)
    os << 'd' << i << ", ";
  if (map.getNumDims() >= 1)
    os << 'd' << map.getNumDims() - 1;
  os << ')';

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
  os << " -> (";
  interleaveComma(map.getResults(),
                  [&](AffineExpr expr) { printAffineExpr(expr); });
  os << ')';
}

void CopyPastedPrintingStuff::printAffineExpr(
    AffineExpr expr, function_ref<void(unsigned, bool)> printValueName) {
  printAffineExprInternal(expr, BindingStrength::Weak, printValueName);
}

void CopyPastedPrintingStuff::printAffineExprInternal(
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