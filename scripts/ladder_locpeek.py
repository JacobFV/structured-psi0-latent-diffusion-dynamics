import json,sys
for f in sys.argv[1:]:
    d=json.load(open(f))["summary"]
    print(f.split("/")[-2], f.split("/")[-1], {k:(round(v,4) if isinstance(v,float) else v) for k,v in d.items() if k not in ("checkpoints","note","robot","bc_label")})
