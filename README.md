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

---

## Regresjon og klassifikasjon

### Lineær regresjon
En modell som predikerer et reelt tall gitt en liste med inputtrekk. Modellen ganger hvert trekk med en vekt og legger til et konstantledd (bias):

```
y = vekt_1 * x_1 + vekt_2 * x_2 + ... + konstantledd
```

Jeg brukte dette på Auto MPG-datasettet for å predikere bensinforbruk (miles per gallon) basert på trekk som vekt, akselerasjon og hestekrefter. Implementert i `models/linear_regression.py`, trent med sklearn sin `LinearRegression`.

### Gjennomsnittlig kvadratfeil (MSE)
Tapsfunksjonen som brukes for lineær regresjon. Måler den gjennomsnittlige kvadratiske forskjellen mellom prediksjonene og de sanne verdiene:

```
MSE = (1/n) * sum((y - y_hat)^2)
```

En lavere MSE betyr bedre prediksjoner. Implementert i `metrics/losses.py`.

### Trekk-seleksjon (feature selection)
Hvilke trekk man bruker har mye å si for hvor bra modellen presterer. Jeg testet ulike kombinasjoner på Auto MPG og fant at å legge til `horsepower` og `model_year` halverte MSE sammenlignet med bare `weight` og `acceleration`. Å legge til enda flere trekk hjalp ikke alltid — noen trekk overlapper med hverandre og bidrar lite nytt.

### Logistisk regresjon
Brukes for klassifikasjon i stedet for regresjon. Fungerer på samme måte som lineær regresjon, men resultatet sendes gjennom en sigmoid-funksjon som gjør om tallet til en sannsynlighet mellom 0 og 1.

Jeg brukte dette på Spambase-datasettet for å klassifisere e-poster som spam eller ikke. Implementert i `models/logistic_regression.py`, trent med sklearn sin `LogisticRegression`.

### Sigmoid-funksjonen
Gjør om et tall fra minus uendelig til pluss uendelig til et tall mellom 0 og 1:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

Høye positive tall gir verdier nærme 1, høye negative tall gir verdier nærme 0.

### Binær kryssentropi (BCE)
Tapsfunksjonen som brukes for logistisk regresjon. Straffer modellen hardt når den er veldig sikker men tar feil, og lite når den er usikker. En lavere BCE betyr bedre prediksjoner. Implementert i `metrics/losses.py`.

En ting jeg lærte her: `log(0)` er udefinert, så prediksjonene må klippes til et lite intervall bort fra 0 og 1 for å unngå numeriske feil.
