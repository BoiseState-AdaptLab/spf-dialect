
module {
  func.func @main() {
		%I = arith.constant 5: index

    "standalone.computation"() ({
			"standalone.bar"(%I) ({
			^bb0():
			%c69 = arith.constant 69 : index
			"standalone.yield"(): () -> ()
			})  {
							reads = [],
							writes = [],
							operand_segment_sizes = dense<[1,0,0,0]> : vector<4xi32>,
							ufNames = [],
							symbolNames = ["I"],
							iteratorTypes = ["reduction"],
							executionSchedule = "{[t,x]->[t,0,x,0]}",
							iterationSpace = "{[t,x] : 1<=t<=T and 1<=x<=X}"
					} : (index) -> ()

			"standalone.bar"(%I) ({
			^bb0():
			%c69 = arith.constant 69 : index
			"standalone.yield"(): () -> ()
			})  {
							reads = [],
							writes = [],
							operand_segment_sizes = dense<[1,0,0,0]> : vector<4xi32>,
							ufNames = [],
							symbolNames = ["I"],
							iteratorTypes = ["reduction"],
							executionSchedule = "{[t,x]->[t,1,x,0]}",
							iterationSpace = "{[t,x] : 1<=t<=T and 1<=x<=X}"
					} : (index) -> ()
		}): () -> ()

		return
	}
}