# Flood Risk Model

Questo repository contiene un modello per la valutazione del rischio di alluvione basato su dati raster.

## Struttura del progetto

```
 flood_risk_model/
 │
 ├── main.py               # Script principale per eseguire il modello
 ├── raster_utils.py       # Funzioni per la gestione e analisi dei dati raster
 ├── hydrology.py          # Modulo per calcoli idrologici e modellazione
 ├── alert_system.py       # Sistema di allerta basato sui risultati del modello
 ├── data/                 # Cartella contenente i dati di input
 │   ├── dem.tif           # Modello digitale di elevazione (DEM)
 │   ├── radar.tif         # Dati radar grezzi
 │   └── reprojected_radar.tif # Dati radar rielaborati e riproiettati
 └── README.md             # Documentazione del progetto
```

## Requisiti

Per eseguire il progetto, assicurati di avere installati i seguenti pacchetti Python:

```bash
pip install -r requirements.txt
```

## Utilizzo

Esegui lo script principale con:

```bash
python main.py
```
