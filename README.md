
### Example Wikidata Queries

```SQL
#defaultView:Table
SELECT DISTINCT ?item ?image ?creatorLabel ?date ?objectLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
  ?item wdt:P276 wd:Q371803.
  ?item wdt:P18 ?image.

  ?item wdt:P31 wd:Q3305213.

  OPTIONAL { ?item wdt:P170 ?creator. }
  OPTIONAL { ?item wdt:P571 ?date. }
  OPTIONAL { ?item wdt:P31 ?object. }
}
```

```SQL
#defaultView:Table
SELECT DISTINCT ?item ?image ?date ?objectLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
  ?item wdt:P276 wd:Q371803.
  ?item wdt:P18 ?image.

  MINUS { ?item wdt:P31 wd:Q125191. }
  MINUS { ?item wdt:P31 wd:Q192425. }
  MINUS { ?item wdt:P31 wd:Q18965. }
  MINUS { ?item wdt:P31 wd:Q3305213. }

  OPTIONAL { ?item wdt:P571 ?date. }
  OPTIONAL { ?item wdt:P31 ?object. }
}
```

```SQL
#defaultView:Table
SELECT DISTINCT ?qid ?depictsLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }

  BIND(wd:Q10301958 AS ?item).
  ?item wdt:P180 ?depicts .
  OPTIONAL { ?depicts rdfs:label ?depictsLabel FILTER (lang(?depictsLabel) = "en") }
}
```
