import richdem as rd
import numpy as np
import matplotlib.pyplot as plt

def compute_hydrology(dem_array):
    """
        Calcola la pendenza, direzione del flusso e accumulo del flusso
        
        args:
            dem_array(numpy): array contenente il dem
            
        return: 
            pendenza, direzione del flusso e accumulo del flusso
    """
    dem = rd.rdarray(dem_array, no_data = np.nan)
    slope = rd.TerrainAttribute(dem, attrib = 'slope_riserun')
    flow_dir = rd.FlowDirectionD8(dem)
    flow_acc = rd.FlowAccumulation(dem, method = 'D8')
    
    plt.figure(figsize=(15, 4))
    plt.subplots(1, 3, 1)
    plt.title("Pendenza del Terreno")
    plt.imshow(slope, cmap='terrain')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title("Direzione del Flusso")
    plt.imshow(flow_dir, cmap='Blues')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Accumulo di Flusso")
    plt.imshow(np.log1p(flow_acc), cmap='viridis')
    plt.colorbar()
    plt.show()
    
    return slope, flow_dir, flow_acc