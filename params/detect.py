OBJECTS_FAUNA = {
  "cat": 0.25,
  "dog": 0.25,
  "bird": 0.25,
  "horse": 0.25,
  "cow": 0.25,
  "bull": 0.25,
  "ox": 0.25,
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
  "conifer": 0.2,
  "fruit": 0.15,
}

OBJECTS_NATURE = {
  "water": 0.8,
  "pond": 0.8,
  "lake": 0.8,
  "cloud": 0.8,
  "sky": 0.8
}

OBJECTS = [
  OBJECTS_FAUNA,
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
