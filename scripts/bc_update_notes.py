"""Refresh the learning-curve table in research/tracks/baselines.md from scripts/bc_curve_table.py."""
import subprocess
p = 'research/tracks/baselines.md'
s = open(p).read()
tab = subprocess.run(['python3', 'scripts/bc_curve_table.py'], capture_output=True, text=True).stdout
start = '**Learning curve (ladder dev scenes, 30 matched seeds per body; regenerate: `python3 scripts/bc_curve_table.py`)**'
a = s.index(start); b = s.index('\n\n', s.index('|---|', a)) + 2
s = s[:a] + start + '\n' + tab + '\n' + s[b:]
open(p, 'w').write(s)
