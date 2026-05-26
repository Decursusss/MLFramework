from datetime import datetime, timedelta, UTC
from data.UtilsForDataGather import save_csv, fetch, load_config

config = load_config("AlbionDataProject")

#Основные функции
def fetch_current_prices(
        item_id=config['item_id'],
        location=config['locations'],
        quality=config['qualities'],
        columns=config['price_columns'],
        save=True
):
    loc_str = ",".join(location) if location else ""
    qual_str = ",".join(str(q) for q in quality)
    item_str = ",".join(id for id in item_id)

    url = f"{config['base_url']}/api/v2/stats/prices/{item_str}.json"
    if loc_str:
        url += f"?locations={loc_str}&qualities={qual_str}"
    else:
        url += f"?qualities={qual_str}"

    data = fetch(url)
    if not data:
        return []

    rows = []
    for entry in data:
        entry.setdefault("city", entry.get("location", ""))
        entry.setdefault("item_id", item_id)
        rows.append(entry)

    if save:
        file_name = f"{item_id}_prices_{datetime.now().strftime('%Y%m%d')}.csv"
        save_csv(file_name, rows, columns, config["data_dir"])
        directory = config["data_dir"] + "/" + file_name
        return rows, directory

    return rows

def fetch_history(
        item_id=config['item_id'],
        locations=config['locations'],
        quality=config['qualities'],
        time_scale=config["time_scale"],
        days=config["days_back"],
        columns=config["history_columns"],
        save=True
):
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    date_format = "%Y-%m-%d"

    loc_str = ",".join(locations) if locations else ""
    qual_str = ",".join(str(q) for q in quality)
    item_str = ",".join(id for id in item_id)

    url = f"{config['base_url']}/api/v2/stats/history/{item_str}.json?date={start.strftime(date_format)}&end_date={end.strftime(date_format)}&time-scale={time_scale}"
    if loc_str:
        url += f"?locations={loc_str}&qualities={qual_str}"
    else:
        url += f"?qualities={qual_str}"

    data = fetch(url)
    if not data:
        return []

    rows = []
    for entry in data:
        base = {
            "item_id": entry.get("item_id", ""),
            "location": entry.get("location", ""),
            "quality": entry.get("quality", 1),
        }
        for point in entry.get("data", []):
            row = {**base, **point}
            rows.append(row)
    if save:
        file_name = f"{item_id}_history_{datetime.now().strftime('%Y%m%d')}.csv"
        save_csv(file_name, rows, columns, config["data_dir"])
        directory = config["data_dir"] + "/" + file_name
        return rows, directory

    return rows