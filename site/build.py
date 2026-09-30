"""Build the browser demo: inline core.js and the sample pictures into template.html.
Writes index.html (standalone, for GitHub Pages) and artifact.html (page body only)."""
import base64, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from varjoluotain import images as I

presets = {}
for n in ("astronaut", "coffee", "chelsea", "rocket"):
    x = I.sample(n, rows=18, cols=24)
    presets[n] = base64.b64encode((np.clip(x, 0, 1) * 255).round().astype(np.uint8).tobytes()).decode()
t = open(os.path.join(here, "template.html")).read()
core = open(os.path.join(here, "core.js")).read()
body = t.replace("/*CORE*/", core).replace("/*PRESETS*/", json.dumps(presets))
open(os.path.join(here, "artifact.html"), "w").write(body)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
open(os.path.join(here, "index.html"), "w").write(head + body.replace("<div class=\"page\">", "</head>\n<body>\n<div class=\"page\">", 1) + "\n</body>\n</html>\n")
print("wrote index.html and artifact.html", len(body), "bytes")
