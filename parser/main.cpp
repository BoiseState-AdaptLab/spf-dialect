#include "../lib/StandaloneTransforms/Parser.h"

int main() {
  Computation jacobi;
  jacobi.addDataSpace("A", "double*");
  jacobi.addDataSpace("B", "double*");
  jacobi.addStmt(
      new Stmt(/*stmtSourceCode*/ "A(x) = (B(x-1) + B(x) + B(x+1))/3",
               /*iterationSpaceStr*/ "{[t,x] : 1<=t<=ub_T and lb_x<=x<=ub_x}",
               /*executionScheduleStr*/ "{[t,x]->[t,0,x,0]}",
               /*dataReadStrs*/
               {{"B", "{[t,x]->[c]: c=x-1}"},
                {"B", "{[t,x]->[x]}"},
                {"B", "{[t,x]->[c]: c=x+1}"}},
               /*dataWriteStrs*/ {{"A", "{[t,x]->[x]}"}}));
  jacobi.addStmt(
      new Stmt(/*stmtSourceCode*/ "B(x) = (A(x-1) + A(x) + A(x+1))/3",
               /*iterationSpaceStr*/ "{[t,x] : 1<=t<=ub_T and lb_x<=x<=ub_x}",
               "{[t,x]->[t,1,x,0]}",
               /*dataReadStrs*/
               {{"A", "{[t,x]->[c]: c=x-1}"},
                {"A", "{[t,x]->[c,a]: c=x and a=x}"},
                {"A", "{[t,x]->[c]: c=x+1}"}},
               /*dataWriteStrs*/ {{"B", "{[t,x]->[x]}"}}));

  jacobi.addTransformation(0, new Relation("{[a,b,c,d]->[a,0,x,0]:x=c-1}"));
  jacobi.addTransformation(1, new Relation("{[a,b,c,d]->[a,0,c,1]}"));

  std::cout << "codeGen ================================\n";
  std::cout << jacobi.codeGen();

  std::cout << "codeJen ================================\n";
  auto codeJen = jacobi.codeJen();
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