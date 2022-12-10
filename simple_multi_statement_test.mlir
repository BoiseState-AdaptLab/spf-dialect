
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
							executionSchedule = "{[i]->[0,i,0]}",
							iterationSpace = "{[i]: 0<=i<I}"
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
							executionSchedule = "{[i]->[0,i,1]}",
							iterationSpace = "{[i]: 0<=i<I}"
					} : (index) -> ()
		}): () -> ()

		return
	}
}