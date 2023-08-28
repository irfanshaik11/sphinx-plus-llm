Entelligence how to run:

1. Install custom sphinx code:  pip uninstall sphinx && pip install . 

2.  git clean -fd && sphinx-quickstart

3. add these lines to conf.py
```
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# be sure to replace/append to existing exensions list in conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]
```

4. sphinx-apidoc -o docs .

5. click on source & navigate to compat.html, to view ai summary.

