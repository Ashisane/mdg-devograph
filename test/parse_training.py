import json
import re

with open('results/sim_output_v2.txt', 'r', encoding='utf-16le', errors='ignore') as f:
    text = f.read()

# Lines look like: [  1/120] loss=0.551221  best=0.551221  (33.0s, ETA 65:21)
# or maybe they contain gamma values too.
# Let's just find all lines with '] loss='
lines = text.split('\n')
data = []
for line in lines:
    if '] loss=' in line:
        # try to extract everything
        # e.g. [  2/120] loss=0.549666  best=0.549666  w=0.5983  a=0.2580
        # Let's extract key=value pairs or just floats
        match_iter = re.search(r'\[\s*(\d+)/\d+\]', line)
        if practically_match := re.search(r'loss=([\d\.]+)', line):
            entry = {}
            if match_iter:
                entry['iter'] = int(match_iter.group(1))
            entry['loss'] = float(practically_match.group(1))
            
            for param in ['best', 'w', 'a', 'gamma_AB', 'gamma_EMS', 'gamma_P']:
                m = re.search(rf'{param}=([\d\.]+)', line)
                if m:
                    entry[param] = float(m.group(1))
            data.append(entry)

with open('parsed_training.json', 'w') as out:
    json.dump(data, out, indent=2)
