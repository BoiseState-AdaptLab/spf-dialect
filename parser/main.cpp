#include "../lib/StandaloneTransforms/Parser.h"

int main() {
  Computation ttm;
  ttm.addDataSpace("U", "double*");
  ttm.addDataSpace("X", "double*");
  ttm.addDataSpace("Y", "double*");
  ttm.addStmt(new Stmt("Y(z,k) += X(j) * U(r,k)",
                       "{[z,ib,ie,j,r,k] : 0<=z<Mf and ib=UFfptr(z) and "
                       "ie=UFfptr(z+1) and ib<=j<ie and r=UFr(j) and 0<=k<R}",
                       "{[z,ib,ie,j,r,k]->[z,ib,ie,j,r,k]}", {}, {}));

  std::cout << "codeGen ================================\n";
  std::cout << ttm.codeGen();

  std::cout << "codeJen ================================\n";
  auto codeJen = ttm.codeJen();
  std::cout << std::get<0>(codeJen);

  std::cout << "parse ==================================\n";
  auto lexer = mlir::standalone::parser::Lexer(std::move(std::get<0>(codeJen)));
  auto parser = mlir::standalone::parser::Parser(lexer, std::get<1>(codeJen));
  auto ast = parser.parseProgram();
  if (ast) {
    std::cout << ast->dump();
  }
}