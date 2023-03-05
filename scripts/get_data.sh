#!/bin/bash
set -euo pipefail
set -x

# awk script to find maximum value in three columns
CMD="
max1<\$1 || NR==1 { max1=\$1 }
max2<\$2 || NR==1 { max2=\$2 }
max3<\$3 || NR==1 { max3=\$3 }
END { print max1, max2, max3 }
"

# https://github.com/frostt-tensor/frostt-tensor.github.io/issues/14
function add_exteded_frostt_header() {
	TMPFILE=$(mktemp /tmp/XXXXXXXXX.tns)
	# add line for: <mode of matrx> <number non zeros>
	echo 3 "$(wc -l <"$1")" >"$TMPFILE"
	# add line for: <max in first demension> <max in second demension> <max in thrid demension>
	cat "$1" | awk "$CMD" >>"$TMPFILE"
	# add data
	cat "$1" >>"$TMPFILE"
	mv "$TMPFILE" "$1"
}

# create functions for scope to prevent mixing up NELL1 with NELL2 etc
function get_nell2() {
	local NELL2="nell-2.tns"
	if [[ ! -e $NELL2 ]]; then
		wget https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz
		gzip -d "$NELL2.gz"
		add_exteded_frostt_header $NELL2
	fi
}

get_nell2
