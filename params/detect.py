OBJECTS_FAUNA = {
  "person": 0.2,
  "people": 0.2,
  "human": 0.2,
  "animal": 0.8,
  "cat": 0.8,
  "dog": 0.8,
  "bird": 0.8,
  "horse": 0.8,
  "cow": 0.8,
  "bull": 0.8,
  "ox": 0.8,
}

OBJECTS_FLORA = {
  "tree": 0.15,
  "grass": 0.15,
  "shrub": 0.15,
  "bush": 0.15,
  "flower": 0.125,
  "vegetation": 0.15,
  "greenery": 0.15,
}

OBJECTS_TREES = {
  "palm tree": 0.15,
  "coniferous tree": 0.15,
}

OBJECTS_NATURE = {
  "water": 0.8,
  "pond": 0.8,
  "lake": 0.8,
  "cloud": 0.8,
  "sky": 0.8
}

OBJECTS = [
  OBJECTS_FLORA,
  OBJECTS_TREES
]

OBJECT2LABEL = {
  "tree": "vegetation",
  "grass": "vegetation",
  "shrub": "vegetation",
  "bush": "vegetation",
  "flower": "vegetation",
  "greenery": "vegetation",

  "people": "person",
  "human": "person",

  "cat": "animal",
  "dog": "animal",
  "bird": "animal",
  "horse": "animal",
  "cow": "animal",
  "bull": "animal",
  "ox": "animal",

  "pond": "water",
  "lake": "water",
}

OBJS_LABELS_IN = [sorted(o.keys()) for o in OBJECTS]
OBJS_LABELS_OUT = [[OBJECT2LABEL.get(l, l) for l in oli] for oli in OBJS_LABELS_IN]
OBJS_THOLDS = [[OBJECTS[i][k] for k in oli] for i,oli in enumerate(OBJS_LABELS_IN)]
