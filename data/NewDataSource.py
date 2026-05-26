import sys
import json
from pathlib import Path

def create_data_source(name: str):
    base = Path("data/sources") / name

    if base.exists():
        print(f"Folder data/{name} already exists!")
        sys.exit(1)

    base.mkdir(parents=True)

    config = {
        "base_url": "Enter your datasource url",
        "data_dir": f"collected/{name}",
        "columns": ["Setup own columns1", "Setup own columns2"]
    }
    config_path = base / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    script = f'''\
from datetime import datetime, timedelta, UTC
from data.UtilsForDataGather import save_csv, fetch, load_config

config = load_config('{name}')

# Основные функции
def fetch_data(columns=config['columns'], save=True, name='{name}'):
    url = f"{{config['base_url']}}"
    
    data = fetch(url)
    if not data:
        return []
    
    rows = []
    for entry in data:
        rows.append(entry)
    
    if save:
        file_name = f"{{name}}_prices_{{datetime.now().strftime('%Y%m%d')}}.csv"
        save_csv(file_name, rows, columns, config["data_dir"])
        directory = config["data_dir"] + "/" + file_name
        return rows, directory
    
    return rows
'''
    script_path = base / f"{name}.py"
    script_path.write_text(script, encoding="utf-8")
    print(f"New Data source folder '{name}' created!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use: python new_data_source.py <name>")
        sys.exit(1)

    create_data_source(sys.argv[1].strip())