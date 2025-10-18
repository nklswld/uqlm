# setup_structure.py
import os
import shutil

# Zielstruktur
structure = [
    ".vscode",
    "benchmarks",
    "results",
    "scripts",
    "notebooks"
]

files = {
    ".vscode/settings.json": """{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.terminal.activateEnvironment": true,
  "python.envFile": "${workspaceFolder}/.env",
  "jupyter.askForKernelRestart": false,
  "files.exclude": {"**/__pycache__": true}
}""",

    ".vscode/launch.json": """{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run: BlackBox",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/run_blackbox.py",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}""",

    ".vscode/tasks.json": """{
  "version": "2.0.0",
  "tasks": [
    {"label": "Run BlackBox", "type": "shell", "command": "python scripts/run_blackbox.py", "problemMatcher": []}
  ]
}""",

    "benchmarks/prompts_demo.csv": "prompt\nNenne drei Fakten über den Planeten Mars mit Quellenangaben.\nErkläre kurz, was semantische Entropie ist.",
    
    "notebooks/.gitkeep": "",
    "results/.gitkeep": "",
    "scripts/.gitkeep": "",
}

def ensure_structure():
    for folder in structure:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Ordner erstellt (oder existiert bereits): {folder}")
    
    for path, content in files.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📝 Datei erstellt: {path}")
        else:
            print(f"⏩ Datei existiert bereits: {path}")

def move_run_blackbox():
    src = "run_blackbox.py"
    dst = os.path.join("scripts", "run_blackbox.py")

    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"🚀 Datei verschoben: {src} → {dst}")
        else:
            print(f"⚠️ Ziel {dst} existiert bereits – Datei wird nicht überschrieben.")
    else:
        print("ℹ️ Keine Datei 'run_blackbox.py' im Hauptverzeichnis gefunden.")

if __name__ == "__main__":
    print("🔧 Richte Projektstruktur ein...\n")
    ensure_structure()
    move_run_blackbox()
    print("\n✅ Struktur erfolgreich erstellt und Skript verschoben!")
