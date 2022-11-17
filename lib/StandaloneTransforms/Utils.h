#ifndef STANDALONE_STANDALONETRANSFORMS_UTILS_H
#define STANDALONE_STANDALONETRANSFORMS_UTILS_H

#include "iegenlib.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>

std::string unmangleUfName(std::string ufMangledName);
mlir::AffineMap createInverse(Relation *transform, Stmt *s0,
                              mlir::PatternRewriter &rewriter);

#endif // STANDALONE_STANDALONETRANSFORMS_UTILS_H