import json

# load character_info from ../data/big5-1024-persona.json
with open('../data/characters.json', 'r') as f:
    character_info = json.load(f)

alias2character = {}
for k, v in character_info.items():
    for a in v["alias"]:
        alias2character[a] = k
    alias2character[k] = k 
    alias2character[k[:k.rfind('-')]] = k 
