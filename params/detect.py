OBJECTS_FAUNA = {
  "bird": 0.25,
  "dog": 0.25,
  "horse": 0.25,
  "ox": 0.25,
  "painting of human": 0.20,
}

OBJECTS_FLORA = {
  "bush": 0.15,
  "flower": 0.12,
  "fruit": 0.15,
  "grass": 0.15,
  "greenery": 0.15,
  "shrub": 0.15,
  "tree": 0.15,
  "vegetation": 0.15,
}

OBJECTS_TREES = {
  "conifer": 0.21,
  "palm tree": 0.15,
}

OBJECTS_PEOPLE = {
  "human face": 0.15,
  "human hand": 0.15,
  "naked human back": 0.20,
  "naked human breast": 0.20,
  "naked human buttocks": 0.20,
  "naked human torso": 0.20,
}

OBJECTS = [
  OBJECTS_FAUNA,
  OBJECTS_FLORA,
  OBJECTS_TREES,
  OBJECTS_PEOPLE,
]

OBJECT2LABEL = {
  "bush": "flora",
  "flower": "flora",
  "fruit": "flora",
  "grass": "flora",
  "greenery": "flora",
  "shrub": "flora",
  "tree": "flora",
  "vegetation": "flora",

  "conifer": "flora",
  "palm tree": "flora",

  "bird": "fauna",
  "dog": "fauna",
  "horse": "fauna",
  "ox": "fauna",
  "painting of human": "person",

  "human face": "face",
  "human hand": "hand",
  "naked human back": "back",
  "naked human breast": "breast",
  "naked human buttocks": "butt",
  "naked human torso": "torso",
}

OBJS_LABELS_IN = [sorted(o.keys()) for o in OBJECTS]
OBJS_LABELS_OUT = [[OBJECT2LABEL.get(l, l) for l in oli] for oli in OBJS_LABELS_IN]
OBJS_THOLDS = [[OBJECTS[i][k] for k in oli] for i,oli in enumerate(OBJS_LABELS_IN)]
