# spf-opt

A lightly wrapped version of
[mlir-opt](https://llvm.org/docs/CommandGuide/opt.html). `spf-opt` includes the
`convert-spf-to-loops` pass that lowers from the `spf` dialect to other MLIR
dialects (mainly `scf`).