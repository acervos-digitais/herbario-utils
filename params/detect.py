OBJECTS_FAUNA = {
  "bird": 0.25,
  "dog": 0.25,
  "horse": 0.25,
  "ox": 0.25,
}

OBJECTS_FLORA = {
  "bush": 0.15,
  "crops": 0.15,
  "flower": 0.125,
  "fruit": 0.15,
  "grass": 0.15,
  "greenery": 0.15,
  "shrub": 0.15,
  "tree": 0.15,
  "vegetation": 0.15,
}

OBJECTS_TREES = {
  "conifer": 0.205,
  "palm tree": 0.15,
}

OBJECTS = [
  OBJECTS_FAUNA,
  OBJECTS_FLORA,
  OBJECTS_TREES
]

OBJECT2LABEL = {
  "bush": "flora",
  "flower": "flora",
  "fruit": "flora",
  "grass": "flora",
  "greenery": "flora",
  "plantation": "flora",
  "shrub": "flora",
  "tree": "flora",
  "vegetation": "flora",

  "conifer": "flora",
  "palm tree": "flora",

  "bird": "fauna",
  "dog": "fauna",
  "horse": "fauna",
  "ox": "fauna",
}

OBJS_LABELS_IN = [sorted(o.keys()) for o in OBJECTS]
OBJS_LABELS_OUT = [[OBJECT2LABEL.get(l, l) for l in oli] for oli in OBJS_LABELS_IN]
OBJS_THOLDS = [[OBJECTS[i][k] for k in oli] for i,oli in enumerate(OBJS_LABELS_IN)]
