import re

NON_DATE_STRINGS = ["s.d.", "unknown", "não ", "http"]

C2Y = {
  "xxi": 2012,
  "xx": 1950,
  "xix": 1850,
  "xviii": 1750,
  "xvii": 1650,
  "xvi": 1550,
  "xv": 1450,
  "xiv": 1350,
  "xiii": 1250
}

def get_year(date_str, title_str):
  date_str = date_str.strip().lower()
  title_str = title_str.strip().lower()

  years = re.findall(r"[1-2][0-9]{3}", date_str)
  cents = re.findall(r"[1-2][0-9][\-\. ]", date_str)
  decs = re.findall(r"[1-2][0-9]{2}[\-\. ]", date_str)
  roman = re.findall(r"[ivx]{2,8}", date_str)
  from_title = re.findall(r"[1-2][0-9]{3}", title_str)

  if len(years) > 0:
    return int(years[0])
  elif len(cents) > 0:
    return int(cents[0][:2] + "50")
  elif len(decs) > 0:
    return int(decs[0][:3] + "5")
  elif len(roman) > 0:
    return C2Y[roman[0]]
  elif len(from_title) > 0:
    return max([int(y) for y in from_title])
  elif date_str == "" or any(x in date_str for x in NON_DATE_STRINGS):
    return 9999
  else:
    raise ValueError(f"ERROR: Can't parse {date_str}")
