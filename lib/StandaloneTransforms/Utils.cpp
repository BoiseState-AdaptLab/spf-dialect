#include "Utils.h"

// TODO: copying all these strings around is definitely bad.
std::string unmangleUfName(std::string ufMangledName) {
  // This is all a hack, it seems like the omega rename thing in hte
  // computation API ads "_<some number>", so I'm just removing that
  // here. There's definitely some way to compose the problem so
  // that this isn't a problem, and that way would be better.
  auto underscorePos = ufMangledName.find("_");
  return ufMangledName.substr(0, underscorePos);
}

/// TODO: AffineMap has an inverse fuction. Transformations might be applied
/// as AffineMaps. This could all just be replaced with that.
///
/// Here we're creating an inverse relation to map transformed execution
/// schedule tuple back to original iteration space tuple.
///
/// For an example of why we need to do this, assume we have a double for
/// loop:
///   for(int i=0; i<I; i++)
///     for(int j=0; j<J; j++)
///       S0
/// In the polyhedral model this would be:
///   Iteration space: {[i,j]: 0<=i<I && 0<=j<J}
///   Execution schedule: {[i,j] -> [i,j]}
/// Take I = 1 and J=2, generated code will execute the iteration space
/// tuples in the order determined by the execution schedule tuples:
///   | Execution Schedule tuple | Iteration space tuple
///   | [0,0]                    | [0,0]
///   | [0,1]                    | [0,1]
///   | [0,2]                    | [0,2]
///   | [1,0]                    | [1,0]
///   | [1,1]                    | [1,1]
///   | [1,2]                    | [1,2]
/// If we apply a permute transformation: {[i,j] -> [j,i]} the order will
/// change:
///   | Execution Schedule tuple | Iteration space tuple
///   | [0,0]                    | [0,0]
///   | [0,1]                    | [1,0]
///   | [1,0]                    | [0,1]
///   | [1,1]                    | [1,1]
///   | [2,0]                    | [0,2]
///   | [2,1]                    | [1,2]
/// Notice that we have to map from the new execution schedule tuple back to
/// the original iteration space tuple. For example, [2,1] might index
/// something out of bounds if 2 was used for i and 1 for j. When we're
/// writing regular code, variable binding does this job for us:
///   for(int i=0; i<I; i++)
///     for(int j=0; j<J; j++)
///       A[i,j] = B[i,j]
/// Vs.
///   for(int j=0; j<J; j++)
///     for(int i=0; i<I; i++)
///       A[i,j] = B[i,j]
/// will execute the same statements just in a different order. But the
/// polyhedral model's representation is more abstract. Statements only know
/// about iteration space variables and don't understand how those variables
/// will be mapped to actual induction variables. Since codegen creates
/// induction variables that range over the execution schedule, we need to
/// create a inverse function that undoes whatever transformations have taken
/// place between the original iteration space and the execution schedule to
/// ensure we execute statements with the right induction variables.
mlir::AffineMap createInverse(Relation *transform, Stmt *statement,
                              mlir::PatternRewriter &rewriter) {

  // To construct the inverse relation we must construct a LHS and RHS.
  // For the LHS:
  //   since the inverse relation executes on the output of the relation we're
  //   inverting we simply apply the relation we're inverting to its input.
  //   In our example, we would generate the LHS for our permute relation
  //   inverse by applying the original relation to [i,j] and producing:
  //     [j,i]
  auto newIterSpace = transform->Apply(statement->getIterationSpace());
  // The MLIR parser below doesn't like ints in maps like [a, 0, b] -> [a,b],
  // those are common when inverting an execution schedule for example.
  // vOmegaReplacer replaces 0 or other ints with variable names. It presumably
  // does other stuff too, I don't actually know much about it.
  VisitorChangeUFsForOmega *vOmegaReplacer = new VisitorChangeUFsForOmega();
  newIterSpace->acceptVisitor(vOmegaReplacer);
  std::string inverseLHS = newIterSpace->getTupleDecl().toString();

  delete vOmegaReplacer;
  delete newIterSpace;

  // For th RHS:
  //   It's a little trickier, under the hood the relation we're inverting:
  //     {[i,j]->[j,i]}
  //   is really something like:
  //     {[t0,t1]->[t2,t3]: t0=t3 && t1=t2}.
  //   To construct the RHS, we go through each of the variables in the the
  //   order of input to the relation we're inverting (t0, t1) and ask for a
  //   function solving for each variable using using only the variables in
  //   the output arity (t2, t3). Since the output of the relation we're
  //   inverting is the LHS of our inverse function, something that solves for
  //   the input to the relation we're inverting using only those variables is
  //   the RHS of the inverse. For example, when we solve for t0 the equality
  //   t0=t3 allows us to just return t3---t3 is i which is what the RHS
  //   should have given an the input of j from the LHS.
  //
  //   To get variable names that make sense we use the tuple decl for the
  //   relation. The tuple decl contains all variables in the relation in
  //   order. For our permute relation it would be [i,j,j,i]. T3 indexes into
  //   this and returns "i".
  //
  // For our example relation: the LHS computes to [j,i], to get the RHS we:
  //   (1) ask for a function solving for t0 only using the variables
  //       [t2, t3]. We get: t3.
  //   (2) we use the variables in the tuple decl to print
  //       that variable to a string we get: i.
  //   (3) we repeat (1) and (2) for t1 yielding: j.
  // putting that together with the RHS we get:
  //   {[j,i]->[i,j]}.
  std::string inverseRHS;
  TupleDecl decl = transform->getTupleDecl();
  // loop over input to relation we're trying to invert
  for (int inputVar = 0; inputVar < transform->inArity(); inputVar++) {
    // find fuction for variable in input in the output (output is the LHS of
    // function we're inverting, it's the RHS of inverse function).
    Exp *inverseForInputVar = transform->findFunction(
        inputVar, transform->inArity(), transform->arity());

    inverseRHS += inverseForInputVar->prettyPrintString(decl);
    inverseRHS += inputVar == transform->inArity() - 1 ? "" : ", "; // add comma
  }

  // Create a string version and parse into MLIR version. This is a little hacky
  // but whatever.
  std::stringstream ss;
  ss << "affine_map<(" << inverseLHS << ") -> (" << inverseRHS << ")>";
  mlir::AffineMap inverseMap =
      mlir::parseAttribute(ss.str(), rewriter.getContext())
          .cast<mlir::AffineMapAttr>()
          .getValue();
  return inverseMap;
}