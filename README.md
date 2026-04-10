# Læringslogg - Maskinlæring

En logg over ting jeg har lært og jobbet med i dette prosjektet.

---

## Ordvektorer

### Tokenisering
Det første steget er å dele opp tekst i enkeltord, kalt tokens. Den enkleste måten er å splitte på mellomrom:
```
"Jeg liker mat" -> ["Jeg", "liker", "mat"]
```
I koden gjøres dette i `data/preprocessing.py` med funksjonen `tokenize()`.

### Stoppordfiltrering
Stoppord er veldig vanlige ord som "og", "er", "det", "en" som egentlig ikke sier noe om hva en tekst handler om. Tanken er at ved å fjerne dem sitter vi igjen med de ordene som faktisk betyr noe. Dette gjøres i `remove_stopwords()` i `data/preprocessing.py`.

### Vokabular
For å jobbe med tekst matematisk må vi gjøre om ord til tall. Vi lager en ordbok som gir hvert ord en unik indeks:
```
{"mat": 0, "liker": 1, "film": 2, ...}
```
Vi begrenser vokabularet til de N mest frekvente ordene. Se `build_vocab()` i `data/preprocessing.py`.

### Sam-forekomstmatrise
En matrise som teller hvor ofte par av ord opptrer i samme setning. Hvis "film" og "skuespiller" ofte dukker opp sammen vil den tilhørende cellen ha et høyt tall. Ideen er at ord som opptrer i lignende kontekster trolig har lignende betydning. Bygges i `build_cooccurrence_matrix()` i `models/vectorizer.py`.

### Ordvektorer
Hver rad i sam-forekomstmatrisen er en vektor som representerer ett ord. Ord med lignende betydning skal i teorien ende opp med lignende vektorer siden de sam-forekommer med mange av de samme ordene.

### Cosinuslikhet
En måte å måle hvor like to vektorer er, uavhengig av lengden deres. Gir et tall mellom -1 og 1, der 1 betyr veldig like og 0 betyr ingen likhet. Implementert i `metrics/similarity.py`.

```
likhet = (A · B) / (|A| x |B|)
```

---

## Dokumentklassifisering

### Datasett: trenings-, validerings- og testsplitter
Datasettet jeg bruker her er NoReC (Norwegian Review Corpus), en samling norske anmeldelser tagget med kategori som spill, litteratur og restauranter.

Data deles i tre deler:
- **Trening** - brukes til å trene modellen
- **Validering** - brukes til å velge hyperparametere (f.eks. k i k-NN). Modellen trener ikke på dette.
- **Test** - brukes én gang til slutt for å se endelig ytelse. Skal ikke røres underveis.

Poenget er å unngå at vi tilpasser modellen for mye til dataene vi evaluerer på. Se `prepare_data()` i `data/loaders.py`.

### CountVectorizer (bag-of-words)
Representerer hvert dokument som en vektor av ordfrekvenser. Med et vokabular på 5000 ord blir hvert dokument en vektor med 5000 tall, der hvert tall sier hvor mange ganger det ordet dukker opp. Rekkefølgen på ord ignoreres, bare frekvenser teller. Bruker scikit-learn sin `CountVectorizer` med `max_features=5000`.

### TF-IDF (Term Frequency - Inverse Document Frequency)
En måte å forbedre rene ordfrekvenser på. Vekter hvert ord etter hvor informativt det er:
- **TF** - hvor ofte ordet dukker opp i dette dokumentet
- **IDF** - hvor sjeldent ordet er på tvers av alle dokumenter

Ord som dukker opp overalt (som "er", "og") får lav vekt. Ord som er vanlige i ett dokument men sjeldne ellers (som "restaurant" i en restaurantanmeldelse) får høy vekt. I mine tester ga TF-IDF mye bedre resultater enn rå ordfrekvenser. Bruker scikit-learn sin `TfidfTransformer`.

### Dimensjonsreduksjon (SVD)
Dokumentvektorene har 5000 dimensjoner som er umulig å visualisere direkte. Truncated SVD reduserer dem til 2 dimensjoner mens den prøver å bevare så mye struktur som mulig, slik at vi kan plotte dem og se om kategoriene klynger seg sammen. Brukes i `scatter_plot()` i `metrics/visualization.py`.

### k-nærmeste naboer (k-NN)
En ganske enkel klassifikasjonsalgoritme. For å klassifisere et nytt dokument:
1. Finn de k mest like dokumentene i treningssettet
2. La disse k naboene stemme, flertallskategorien vinner

Valg av k har ganske mye å si:
- Liten k (f.eks. k=1) - sensitiv for enkelteksempler, kan føre til overtilpasning
- Stor k (f.eks. k=5000) - domineres av majoritetsklassen, undertilpasser

I mine eksperimenter ga k=20 med TF-IDF best resultater på valideringssettet. Implementert i `models/classifier.py`.

### Overtilpasning og undertilpasning
- **Overtilpasning** - modellen memorerer treningsdata og generaliserer dårlig til nye data
- **Undertilpasning** - modellen er for enkel til å fange opp mønstre i dataene

### Evalueringsmetrikker

**Nøyaktighet** - andelen korrekte prediksjoner totalt. Enkel å forstå, men kan være misvisende når klassene er skjevt fordelt. Siden ca. 66% av NoReC-dataene er litteratur ville en modell som alltid gjetter litteratur allerede fått 66% nøyaktighet.

**F1-mål** - gjennomsnittet av presisjon og sensitivitet. Litt mer robust når klassene er ujevnt fordelt.

**Presisjon** - av alle dokumenter vi predikerte som klasse X, hvor mange var faktisk X?

**Sensitivitet** - av alle dokumenter som faktisk er klasse X, hvor mange predikerte vi riktig?
