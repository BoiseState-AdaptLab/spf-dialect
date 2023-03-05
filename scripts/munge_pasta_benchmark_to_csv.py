import sys

def process(i, lines, search_term, platform, benchmark):
	filename = lines[i].split("/")[-1].strip()
	i += 1

	while i < len(lines):
		line = lines[i]
		if line.startswith("input file"):
			break

		# looking for something that looks like: "{search_term}: 0.006577440 s
		if line.startswith(f"{search_term}:"):
			seconds = int(1000 * float(line.split(":")[1].split(" ")[1]))
			print(f"{platform}, {benchmark}, pasta, {filename}, {seconds}")

		i += 1

	return i

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(f"usage: python3 {sys.argv[0]} pasta-file-to-munge")
		sys.exit()

	with open(sys.argv[1], 'r') as file:
		lines = file.readlines()
		i = 0
		# process until first file
		while i < len(lines):
			if lines[i].startswith("input file"):
				break
			i+= 1

		# expecting the benchmarks to have been run like:
		# 	for file in files:
		# 		cpu mttkrp
		# 		gpu mttkrp
		# 		cpu ttm
		# 		gpu ttm
		while i < len(lines):
			i = process(i, lines, "[Cpu SpTns MTTKRP]", "cpu", "mttkrp")
			i = process(i, lines, "[Cuda SpTns MTTKRP]", "gpu", "mttkrp")
			i = process(i, lines, "[Cpu SpTns * Mtx]", "cpu", "ttm")
			i = process(i, lines, "[Cuda SpTns * Mtx]", "gpu", "ttm")