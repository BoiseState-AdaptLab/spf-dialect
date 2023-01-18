#include "../lib/StandaloneTransforms/Parser.h"

int main() {
  Computation ttm;
  ttm.addDataSpace("U", "double*");
  ttm.addDataSpace("X", "double*");
  ttm.addDataSpace("Y", "double*");
  ttm.addStmt(new Stmt("Y(z,k) += X(j) * U(r,k)",
                       "{[z,j,r,k] : 0<=z<Mf and UFfptr(z)<=j<UFfptr(z+1) and r=UFr(j) and 0<=k<R}",
                       "{[z,j,r,k]->[z,j,r,k]}", {}, {}));

  std::cout << "codeGen ================================\n";
  std::cout << ttm.codeGen();

  std::cout << "codeJen ================================\n";
  auto codeJen = ttm.codeJen();
  auto code = std::get<0>(codeJen);
  auto vOmegaReplacer = std::get<1>(codeJen);
  std::cout << code;

  std::cout << "parse ==================================\n";
  auto lexer = mlir::standalone::parser::Lexer(std::move(code));
  auto parser = mlir::standalone::parser::Parser(lexer, vOmegaReplacer);
  auto ast = parser.parseProgram();
  if (ast) {
    std::cout << ast->dump();
  }

  delete vOmegaReplacer;
}