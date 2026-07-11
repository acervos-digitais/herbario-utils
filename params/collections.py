MUSEUMS = {
  "museu_paulista": {
    "label": "Museu Paulista",
    "dir": "MuseuPaulista",
    "file": "museu_paulista",
    "type": "wikidata",
    "collection": True,
    "objects": ["painting", "drawing"]
  },
  "masp": {
    "label": "MASP",
    "dir": "MASP",
    "file": "masp",
    "type": "wikidata",
    "collection": False,
    "objects": ["painting", "drawing"]
  },
  "pinacoteca_sp": {
    "label": "Pinacoteca de São Paulo",
    "dir": "PinacotecaSP",
    "file": "pinacoteca_sp",
    "type": "wikidata",
    "collection": True,
    "objects": ["painting", "drawing"]
  },
  "museu_historico": {
    "label": "Museu Histórico Nacional",
    "dir": "MuseuHistorico",
    "file": "museu_historico",
    "type": "wikidata",
    "collection": True,
    "objects": ["painting", "drawing"]
  },
  "hercules_florence": {
    "label": "Instituto Hércules Florence",
    "dir": "HerculesFlorence",
    "file": "hercules_florence",
    "type": "wikidata",
    "collection": True,
    "objects": ["watercolor painting", "illustration"]
  },
  "belas_artes": {
    "label": "Museu Nacional de Belas Artes",
    "dir": "BelasArtes",
    "file": "belas_artes",
    "type": "wikidata",
    "collection": False,
    "objects": ["painting", "drawing"]
  },
  "itau_brasiliana": {
    "label": "Coleção Brasiliana Itaú",
    "dir": "ItauBrasiliana",
    "file": "itau_brasiliana",
    "type": "wikidata",
    "collection": True,
    "objects": ["painting", "drawing"]
  },
  "brasiliana": {
    "label": "Brasiliana",
    "dir": "Brasiliana",
    "file": "brasiliana",
    "type": "tainacan",
    "collection": False,
    "objects": ["painting", "drawing"]
  },
  "file": {
    "label": "FILE",
    "dir": "File",
    "file": "file",
    "type": "tainacan",
    "collection": False,
    "objects_all": [
      "animation", "architecture", "algorithmic art", "public art", "robotic art", "synthetic art", "sound art", "bio art", "cinema", "dance", "digital photography", "fractal art", "games", "generative art", "art installation", "internet art", "digital language", "art mapping", "mobile art", "performace", "digital poetry", "software art", "artificial life", "video art"
    ],

    "objects_huge_>500": [ "animation", "video art" ],

    "objects": [
      "architecture", "algorithmic art", "robotic art", "bio art", "dance",
      "digital photography", "fractal art", "art mapping", "mobile art", "performace",
      "digital poetry", "software art", "artificial life",
      "public art", "synthetic art", "cinema", "generative art", "digital language",
      "sound art", "games", "art installation", "internet art"
    ]
  },
  "macusp": {
    "label": "MAC USP",
    "dir": "MACUSP",
    "file": "macusp",
    "type": "macusp",
    "collection": False,
    "objects": ["painting", "drawing"],
    "path": "./metadata/csv/macusp.csv"
  },
}
