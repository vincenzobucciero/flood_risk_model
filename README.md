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

## Guida all'installazione ed esecuzione

Per configurare l'ambiente ed installare tutte le dipendenze necessarie.

1. Clona il repository
   
   ```bash
   git clone https://github.com/checcafor/flood_risk_model
   cd flood_risk_model
   ```
2. Configura l'ambiente virtuale
   > **ℹ NOTE**
   > La libreria `RichDEM` richiede funzionalità specifiche di `Python 3.10` per funzionare correttamente. Altre versioni di Python potrebbero non supportare alcune dipendenze o funzionalità utilizzate dalla libreria.
   
   ```bash
   # rimuovi l'ambiente virtuale esistente (se presente)
   rm -rf venv

   # crea un nuovo ambiente virtuale con Python 3.10
   python3.10 -m venv venv

   # attiva l'ambiente virtuale
   source venv/bin/activate # su linux/mac
   ``` 
4. Installare le altre dipendenze
   
   ```bash
   pip install -r requirements.txt
   pip install richdem
   ```
5. Esecuzione del Modello
   
   ```bash
   python main.py
   ```
