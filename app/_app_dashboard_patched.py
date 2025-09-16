# Auto-shim: redirige a app_dashboard.py (mantiene compat para referencias antiguas)
import sys, runpy, pathlib
base = pathlib.Path(_file_).parent
sys.path.insert(0, str(base))
runpy.run_path(str(base / "app_dashboard.py"), run_name="_main_")
