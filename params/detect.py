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


class DinoObjects:
  OBJECTS = {
    "bird": {
      "terms": [
        "bird",
        "fowl",
        "flying bird",
        "perched bird",
        "waterfowl",
        "game bird",
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
      "threshold": 0.20,
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
      "threshold": 0.20,
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
      "threshold": 0.20,
    },

    "human": {
      "terms": [
        "person",
        "human",
        "man",
        "woman",
        "child",
        "people",
        "human figure",
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
      "threshold": 0.20,
    },

    "palm tree": {
      "terms": [
        "palm tree",
        "palm",
        "palm trees",
        "date palm",
        "tropical tree",
        "coconut palm",
        "palm grove"
      ],
      "threshold": 0.20,
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
      "threshold": 0.20,
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
      "threshold": 0.20,
    },

    "fruits": {
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
      "threshold": 0.20,
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
      "threshold": 0.20,
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
      "threshold": 0.20,
    },

    "hand": {
      "terms": [
        "hand",
        "hands",
         "human hand",
         "fingers",
         "palm",
         "glove",
         "gloves"
      ],
      "threshold": 0.20,
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
      "threshold": 0.20,
    },

    "torso": {
      "terms": [
        "exposed torso",
        "bare torso",
        "naked chest",
        "bare chest",
        "naked torso",
      ],
      "threshold": 0.30,
    },

    "back": {
      "terms": [
        "bare back",
        "exposed back",
        "naked back",
        "exposed shoulders",
      ],
      "threshold": 0.30,
    },
  }
