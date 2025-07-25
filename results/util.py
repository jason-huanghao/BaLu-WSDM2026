import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ghibli import *

def create_grouped_boxplot(df, save_fn, metric='PEHE', figsize=(5, 8), palette_name='PonyoMedium'):
    """
    Create a grouped box plot for the specified metric with beautiful Ghibli colors
    
    Parameters:
    df: pandas DataFrame with columns 'Method', 'Dataset', 'MAE', 'PEHE'
    metric: str, either 'MAE' or 'PEHE'
    figsize: tuple, figure size (width, height)
    palette_name: str, Ghibli palette name (e.g., 'PonyoMedium', 'LaputaMedium', 'SpiritedMedium')
    """
    # set_palette(palette_name)
    custom_colors =[
    '#E63946',  # Vibrant red
    '#F77F00',  # Bright orange  
    '#FCBF49',  # Golden yellow
    '#06D6A0',  # Mint green
    '#118AB2',  # Ocean blue
    '#073B4C',  # Dark navy
    '#8E44AD',  # Rich purple
    '#E91E63',  # Pink
    '#795548',  # Warm brown
    '#607D8B'   # Blue gray
]
    custom_colors = [
    '#D32F2F',  # Strong red
    '#FF6F00',  # Vivid orange
    '#FBC02D',  # Bright yellow
    '#388E3C',  # Forest green
    '#00ACC1',  # Cyan
    '#1976D2',  # Blue
    '#7B1FA2',  # Purple
    '#C2185B',  # Deep pink
    '#5D4037',  # Brown
    '#455A64'   # Dark gray
]
    custom_colors = [
    '#DC143C',  # Crimson
    '#FF8C00',  # Dark orange
    '#FFD700',  # Gold
    '#32CD32',  # Lime green
    '#00CED1',  # Dark turquoise
    '#4169E1',  # Royal blue
    '#9932CC',  # Dark orchid
    '#FF1493',  # Deep pink
    '#8B4513',  # Saddle brown
    '#2F4F4F'   # Dark slate gray
]
    custom_colors = [
    '#E74C3C',  # Strong red
    # '#E63946',
    '#FF6F00',
    # '#F39C12',  # Vivid orange
    '#F1C40F',  # Bright yellow
    # '#27AE60',  # Strong green
    '#388E3C',
    # '#32CD32',
    '#3498DB',  # Clear blue
    '#9B59B6',  # Rich purple
    '#E91E63',  # Magenta
    '#1ABC9C',  # Turquoise
    '#95A5A6',  # Silver
    '#607D8B'
    # '#8B4513'
    # '#34495E'   # Dark blue gray
]
    
    if metric not in ['MAE', 'PEHE']:
        raise ValueError("Metric must be either 'MAE' or 'PEHE'")
    
    # Filter out rows where the metric is NaN
    df_filtered = df.dropna(subset=[metric])
    
    df_filtered = df_filtered.sort_values(['Dataset', 'Method'], ascending=[False, True])
    
    if df_filtered.empty:
        print(f"No data available for metric: {metric}")
        return
    
    # Get number of unique methods for color palette
    n_methods = df_filtered['Method'].nunique()
    
    # Set style for better aesthetics
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    max_value = df_filtered[metric].max()
    # max_value= df_filtered[metric].nlargest(2).iloc[1]
    min_value = df_filtered[metric].min()
    buffer = (max_value - min_value) * 0.02
    ax.set_ylim(min_value - buffer, max_value + buffer)
    
    # Create the box plot with beautiful Ghibli colors
    box_plot = sns.boxplot(data=df_filtered, x='Dataset', y=metric, hue='Method', ax=ax, palette=custom_colors)#, order=['Youtube' 'Flickr' 'BlogCatalog' 'Instagram'])
    
    # Enhance box plot aesthetics
    for patch in box_plot.artists:
        patch.set_alpha(0.8)  # Add slight transparency
    
    # Customize the plot with elegant styling
    # ax.set_title(f'{metric} Distribution by Dataset and Method',
    #             fontsize=18, fontweight='bold', pad=20,
    #             color='#2F4F4F')  # Dark slate gray
    ax.set_xlabel('Dataset', fontsize=20, fontweight='medium', color='#2F4F4F')
    ax.set_ylabel(metric, fontsize=20, fontweight='medium', color='#2F4F4F')
    
    # ax.set_xlim(-0.25, n_datasets-0.75)

    # Style the axes
    ax.tick_params(axis='x', rotation=0, labelsize=18, colors='#2F4F4F')
    ax.tick_params(axis='y', labelsize=18, colors='#2F4F4F')
    
    # Enhance legend - positioned at bottom with 2 rows and 5 columns
    legend = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=15,
                      frameon=True, shadow=True, fancybox=True, ncol=5)
    legend.get_frame().set_facecolor('#FFFEF7')  # Cream background
    legend.get_frame().set_alpha(0.8)
    
    # Add elegant grid
    # ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#8FBC8F')
    ax.set_facecolor('#FFFEF7')  # Cream background
    
    # Set spine colors
    # for spine in ax.spines.values():
    #     spine.set_color('#8FBC8F')
    #     spine.set_linewidth(1.2)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.25, top=0.9)  # Increase bottom margin for legend
    
    plt.tight_layout()
    plt.savefig(save_fn, bbox_inches='tight')
    plt.show()

def create_grouped_boxplot1(df, save_fn, metric='PEHE', figsize=(8, 8), palette_name='PonyoMedium'):
    """
    Create a grouped box plot for the specified metric with beautiful Ghibli colors
    
    Parameters:
    df: pandas DataFrame with columns 'Method', 'Dataset', 'MAE', 'PEHE'
    metric: str, either 'MAE' or 'PEHE'
    figsize: tuple, figure size (width, height)
    palette_name: str, Ghibli palette name (e.g., 'PonyoMedium', 'LaputaMedium', 'SpiritedMedium')
    """
    set_palette(palette_name)
    if metric not in ['MAE', 'PEHE']:
        raise ValueError("Metric must be either 'MAE' or 'PEHE'")
    
    # Filter out rows where the metric is NaN
    df_filtered = df.dropna(subset=[metric])

    df_filtered = df_filtered.sort_values(['Dataset', 'Method'], ascending=[False, True])
    
    if df_filtered.empty:
        print(f"No data available for metric: {metric}")
        return
    
    # Set style for better aesthetics
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    max_value = df_filtered[metric].max()
    # max_value= df_filtered[metric].nlargest(2).iloc[1]
    min_value = df_filtered[metric].min()
    buffer = (max_value - min_value) * 0.02
    ax.set_ylim(min_value - buffer, max_value + buffer)
   

    # Create the box plot with beautiful Ghibli colors
    box_plot = sns.boxplot(data=df_filtered, x='Dataset', y=metric, hue='Method',  ax=ax)#, order=['Youtube' 'Flickr' 'BlogCatalog' 'Instagram'])
    
    # Enhance box plot aesthetics
    for patch in box_plot.artists:
        patch.set_alpha(0.8)  # Add slight transparency
    
    # Customize the plot with elegant styling
    # ax.set_title(f'{metric} Distribution by Dataset and Method', 
    #             fontsize=18, fontweight='bold', pad=20, 
    #             color='#2F4F4F')  # Dark slate gray
    ax.set_xlabel('Dataset', fontsize=20, fontweight='medium', color='#2F4F4F')
    ax.set_ylabel(metric, fontsize=20, fontweight='medium', color='#2F4F4F')
    
    # Style the axes
    ax.tick_params(axis='x', rotation=0, labelsize=18, colors='#2F4F4F')
    ax.tick_params(axis='y', labelsize=18, colors='#2F4F4F')
    
    # Enhance legend
    legend = ax.legend(bbox_to_anchor=(0.0, 1), loc='upper left', fontsize=15,
                      frameon=True, shadow=True, fancybox=True)
    legend.get_frame().set_facecolor('#FFFEF7')  # Cream background
    legend.get_frame().set_alpha(0.8)
    
    # Add elegant grid
    # ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='#8FBC8F')
    ax.set_facecolor('#FFFEF7')  # Cream background
    # ax.margins(x=0.05)
    # Set spine colors
    # for spine in ax.spines.values():
    #     spine.set_color('#8FBC8F')
    #     spine.set_linewidth(1.2)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)  # Reduce margins
    
    plt.tight_layout()
    plt.savefig(save_fn, bbox_inches='tight')
    plt.show()

def create_comparison_plots(df, palette_name='PonyoMedium'):
    """
    Create side-by-side box plots for both MAE and PEHE metrics with beautiful Ghibli styling
    
    Parameters:
    df: pandas DataFrame with columns 'Method', 'Dataset', 'MAE', 'PEHE'
    palette_name: str, Ghibli palette name (e.g., 'PonyoMedium', 'LaputaMedium', 'SpiritedMedium')
    """
    set_palette(palette_name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor('#FFFEF7')  # Cream background for the entire figure
    
    # Filter data for each metric
    df_mae = df.dropna(subset=['MAE'])
    df_pehe = df.dropna(subset=['PEHE'])
    
    # Get colors for both plots
    n_methods = max(df_mae['Method'].nunique() if not df_mae.empty else 0,
                   df_pehe['Method'].nunique() if not df_pehe.empty else 0)
    
    # MAE plot
    if not df_mae.empty:
        box1 = sns.boxplot(data=df_mae, x='Dataset', y='MAE', hue='Method',  ax=ax1)
        
        # Enhance aesthetics
        for patch in box1.artists:
            patch.set_alpha(0.8)
        
        ax1.set_title('MAE Distribution by Dataset and Method', 
                     fontsize=16, fontweight='bold', pad=15, color='#2F4F4F')
        ax1.set_xlabel('Dataset', fontsize=12, fontweight='medium', color='#2F4F4F')
        ax1.set_ylabel('MAE', fontsize=12, fontweight='medium', color='#2F4F4F')
        ax1.tick_params(axis='x', rotation=45, labelsize=10, colors='#2F4F4F')
        ax1.tick_params(axis='y', labelsize=10, colors='#2F4F4F')
        # ax1.grid(True, alpha=0.3, axis='y', linestyle='--', color='#8FBC8F')
        ax1.set_facecolor('#FFFEF7')
        
        # Style spines
        for spine in ax1.spines.values():
            spine.set_color('#8FBC8F')
            spine.set_linewidth(1.2)
        
        legend1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                           frameon=True, shadow=True, fancybox=True)
        legend1.get_frame().set_facecolor('#FFFEF7')
        legend1.get_frame().set_alpha(0.9)
    
    # PEHE plot
    if not df_pehe.empty:
        box2 = sns.boxplot(data=df_pehe, x='Dataset', y='PEHE', hue='Method', ax=ax2)
        
        # Enhance aesthetics
        for patch in box2.artists:
            patch.set_alpha(0.8)
        
        ax2.set_title('PEHE Distribution by Dataset and Method', 
                     fontsize=16, fontweight='bold', pad=15, color='#2F4F4F')
        ax2.set_xlabel('Dataset', fontsize=12, fontweight='medium', color='#2F4F4F')
        ax2.set_ylabel('PEHE', fontsize=12, fontweight='medium', color='#2F4F4F')
        ax2.tick_params(axis='x', rotation=45, labelsize=10, colors='#2F4F4F')
        ax2.tick_params(axis='y', labelsize=10, colors='#2F4F4F')
        # ax2.grid(True, alpha=0.3, axis='y', linestyle='--', color='#8FBC8F')
        ax2.set_facecolor('#FFFEF7')
        
        # Style spines
        # for spine in ax2.spines.values():
        #     spine.set_color('#8FBC8F')
        #     spine.set_linewidth(1.2)
        
        legend2 = ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                           frameon=True, shadow=True, fancybox=True)
        legend2.get_frame().set_facecolor('#FFFEF7')
        legend2.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    plt.show()

def print_summary_stats(df, metric):
    """
    Print summary statistics for the specified metric grouped by method and dataset
    """
    if metric not in ['MAE', 'PEHE']:
        raise ValueError("Metric must be either 'MAE' or 'PEHE'")
    
    df_filtered = df.dropna(subset=[metric])
    
    if df_filtered.empty:
        print(f"No data available for metric: {metric}")
        return
    
    print(f"\n=== Summary Statistics for {metric} ===")
    summary = df_filtered.groupby(['Dataset', 'Method'])[metric].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary.round(4))

# Usage examples:

# Optional: Install ghibli package for authentic colors
# pip install ghibli


# 1. Show the beautiful color palette
# show_color_palette()

# # 2. Create box plot for PEHE metric with Ghibli colors
# create_grouped_boxplot(result_df, metric='PEHE')

# # 3. Create box plot for MAE metric with Ghibli colors
# create_grouped_boxplot(result_df, metric='MAE')

# # 4. Create side-by-side comparison of both metrics with Ghibli styling
# create_comparison_plots(result_df)

# # 4. Print summary statistics
# print_summary_stats(result_df, 'PEHE')
# print_summary_stats(result_df, 'MAE')

# 5. Interactive selection (uncomment to use)
# def interactive_plot():
#     metric = input("Enter metric to visualize (MAE or PEHE): ").upper()
#     if metric in ['MAE', 'PEHE']:
#         create_grouped_boxplot(result_df, metric=metric)
#     else:
#         print("Invalid metric. Please enter 'MAE' or 'PEHE'")
# 
# interactive_plot()