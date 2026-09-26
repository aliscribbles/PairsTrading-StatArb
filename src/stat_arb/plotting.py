import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches import Rectangle
from typing import Optional,Dict,Tuple


#The Visualization Module
def plot_correlation_matrix(returns: pd.DataFrame, sector_groups: Optional[Dict]=None,
                            figsize:Tuple=(12,9)):
    
    
    """Plots a correlation heatmap with optional sector bounding boxes"""
    
    corr=returns.corr()
    
    fig,ax=plt.subplots(figsize=figsize)
    
    #Create heatmap
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1,vmax=1)
    #Set ticks and labels for axes
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    #Pass the heatmap object to color bar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if sector_groups:
        # Create a mapping of ticker to its integer index position in the matrix
        idx_map = {ticker: i for i, ticker in enumerate(corr.columns)}
        
        for sector_name, tickers in sector_groups.items():
            # Find the positions of the tickers in this sector
            positions = [idx_map[t] for t in tickers if t in idx_map]
            
            if positions:
                # Find the min and max index to draw the box
                lo, hi = min(positions), max(positions)
                
                # Draw the rectangle
                rect = Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                                 linewidth=2, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                
    ax.set_title('Return Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
        
    return fig
        
