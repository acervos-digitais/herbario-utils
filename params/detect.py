class Owlv2Objects:
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
    "human foot": 0.15,
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

# Outside after class definition so comprehensions work
_OBJECTS = Owlv2Objects.OBJECTS
_OBJECT_LABELS_OUT = sorted([l for ls in _OBJECTS for l in ls.keys()])
_OBJECT_LABELS_IN = [sorted(o.keys()) for o in _OBJECTS]
_OBJECT_THOLDS = [[_OBJECTS[i][k] for k in oli] for i,oli in enumerate(_OBJECT_LABELS_IN)]
Owlv2Objects.OBJECT_LABELS_OUT = _OBJECT_LABELS_OUT
Owlv2Objects.OBJECT_LABELS_IN = _OBJECT_LABELS_IN
Owlv2Objects.OBJECT_THOLDS = _OBJECT_THOLDS

class DinoObjects:
  OBJECTS = {
    "bird": {
      "terms": [
        "bird",
        "fowl",
        "flying bird",
        "perched bird",
        "waterfowl",
      ],
      "threshold": 0.2,
    },

    "dog": {
      "terms": [
        "dog",
        "canine",
        "hound",
        "puppy",
        "mongrel",
        "domestic dog",
      ],
      "threshold": 0.15
    },

    "horse": {
      "terms": [
        "horse",
        "equine",
        "steed",
        "stallion",
        "mare",
        "pony",
      ],
      "threshold": 0.125
    },

    "ox": {
      "terms": [
        "ox",
        "oxen",
        "bovine",
        "cattle",
        "bull",
        "cow",
        "draft animal",
      ],
      "threshold": 0.125
    },

    "fish": {
      "terms": [
        "fish",
        "fishes",
        "aquatic animal"
        "caught fish",
        "whole fish",
        "dead fish",
      ],
      "threshold": 0.175
    },

    "human figure": {
      "terms": [
        "person",
        "human",
        "man",
        "woman",
        "child",
        "people",
        "human figure",
        "angel",
        "saint",
      ],
      "threshold": 0.20,
    },

    "tree": {
      "terms": [
        "tree",
        "tree trunk",
        "foliage",
        "plant",
        "grove",
        "oak tree",
      ],
      "threshold": 0.15
    },

    "palm tree": {
      "terms": [
        "palm tree",
        "palm trees",
        "date palm",
        "tropical tree",
        "coconut palm",
        "palm grove"
      ],
      "threshold": 0.15
    },

    "conifer": {
      "terms": [
        "conifer",
        "pine tree",
        "pine",
        "evergreen tree",
        "fir tree",
        "spruce tree",
      ],
      "threshold": 0.15
    },

    "flower": {
      "terms": [
        "flower",
        "flowers",
        "wildflower",
        "blossom",
        "bloom",
        "floral",
        "bouquet",
        "floral arrangement",
      ],
      "threshold": 0.15
    },

    "fruit": {
      "terms": [
        "fruit",
        "fruits",
        "produce",
        "fresh fruit",
        "fruit basket",
        "apple",
        "pear",
        "citrus",
      ],
      "threshold": 0.15
    },

    "shrubbery": {
      "terms": [
        "shrub",
        "shrubbery",
        "bush",
        "bushes",
        "hedge",
        "thicket",
      ],
      "threshold": 0.15
    },

    "face": {
      "terms": [
        "face",
        "human face",
        "facial features",
        "mask",
        "face mask",
        "masquerade mask",
      ],
      "threshold": 0.15
    },

    "hand": {
      "terms": [
        "hand",
        "hands",
        "human hand",
        "fingers",
        "glove",
        "gloves"
      ],
      "threshold": 0.15
    },

    "foot": {
      "terms": [
        "foot",
        "feet",
        "human foot",
        "shoe",
        "footwear",
        "boot",
      ],
      "threshold": 0.15
    },

    "bare torso": {
      "terms": [
        "exposed torso",
        "bare torso",
        "naked torso",
      ],
      "threshold": 0.30,
    },

    "bare buttocks": {
      "terms": [
        "bare buttocks",
        "exposed buttocks",
        "naked buttocks",
        "naked butt",
        "exposed butt",
      ],
      "threshold": 0.40,
    },

    "bare breast": {
      "terms": [
        "bare breast",
        "exposed breast",
        "naked breast",
        "breast",
        "breasts",
        "nipple",
      ],
      "threshold": 0.30,
    },
  }

# Outside after class definition so comprehensions work
_OBJECTS = DinoObjects.OBJECTS
_OBJECT_LABELS_OUT = sorted(list(_OBJECTS.keys()))
_OBJECT_LABELS_IN = [_OBJECTS[l]["terms"] for l in _OBJECT_LABELS_OUT]
_OBJECT_THOLDS = [[_OBJECTS[l]["threshold"] for t in _OBJECTS[l]["terms"]] for l in _OBJECT_LABELS_OUT]
_OBJECT_LABELS_IN2OUT = { lin: lout for lout in _OBJECT_LABELS_OUT for lin in _OBJECTS[lout]["terms"] }

DinoObjects.OBJECT_LABELS_OUT = _OBJECT_LABELS_OUT
DinoObjects.OBJECT_LABELS_IN = _OBJECT_LABELS_IN
DinoObjects.OBJECT_THOLDS = _OBJECT_THOLDS
DinoObjects.OBJECT_LABELS_IN2OUT = _OBJECT_LABELS_IN2OUT
