import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_results(result_df, hyper="$\\beta$", metric='PEHE', confidence_interval=True, save_pdf=None):
    """
    Create subplots for each Dataset showing line plots with confidence intervals.
    Each subplot shows lines for different p_miss values.
    Points represent mean values and confidence intervals represent standard deviation.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame containing columns: Dataset, hyperparameter column, $p_{\\text{miss}}, and metric column
    hyper : str
        Name of the hyperparameter column (e.g., "$\\beta$", "$\\eta$", "$\\gamma$")
    metric : str  
        Name of the metric column (e.g., 'PEHE', 'MAE', 'SUM')
    confidence_interval : bool
        Whether to show confidence intervals based on standard deviation
    save_pdf : str or None
        Path to save the plot as PDF file
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Define mapping and order for hyperparameter values
    value_mapping = {
        0.0: '0.0',
        0.0001: '$10^{-4}$',
        0.001: '$10^{-3}$', 
        0.01: '$10^{-2}$',
        0.1: '$10^{-1}$',
        '0.0': '0.0',
        '0.0001': '$10^{-4}$',
        '0.001': '$10^{-3}$', 
        '0.01': '$10^{-2}$',
        '0.1': '$10^{-1}$'
    }
    desired_order = ['0.0', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
    dataset_order = ['Instagram', 'Youtube', 'BlogCatalog', 'Flickr']
    
    # Prepare data
    df = result_df.copy()
    
    # Map hyperparameter values to display format
    df['hyper_display'] = df[hyper].map(value_mapping)
    
    # Handle any unmapped values by converting to string
    df['hyper_display'] = df['hyper_display'].fillna(df[hyper].astype(str))
    
    # Calculate mean, std, and count for each combination of Dataset, hyperparameter, and p_miss
    grouped = df.groupby(['Dataset', 'hyper_display', '$p_{\\text{miss}}$'])[metric].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Fill NaN std values with 0 (for cases with only one measurement)
    grouped['std'] = grouped['std'].fillna(0)
    
    # Get datasets in desired order (only those present in data)
    available_datasets = grouped['Dataset'].unique()
    datasets = [d for d in dataset_order if d in available_datasets]
    
    # If no datasets match the predefined order, use all available datasets
    if not datasets:
        datasets = sorted(available_datasets)
    
    # Get p_miss values
    p_miss_values = sorted(grouped['$p_{\\text{miss}}$'].unique())
    
    # Create figure
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5*len(datasets), 2.5))
    if len(datasets) == 1:
        axes = [axes]
    
    # Colors and styles
    colors = sns.color_palette("viridis", len(p_miss_values))
    line_styles = ['-', '--', '-.', ':']
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_data = grouped[grouped['Dataset'] == dataset]
        
        for j, p_miss in enumerate(p_miss_values):
            # Get data for this p_miss value
            pmiss_data = dataset_data[dataset_data['$p_{\\text{miss}}$'] == p_miss].copy()
            
            if len(pmiss_data) == 0:
                continue
            
            # Prepare data for plotting
            plot_data = []
            for _, row in pmiss_data.iterrows():
                if row['hyper_display'] in desired_order:
                    x_pos = desired_order.index(row['hyper_display'])
                    plot_data.append({
                        'x_pos': x_pos,
                        'y_mean': row['mean'],
                        'y_std': row['std'],
                        'count': row['count'],
                        'x_label': row['hyper_display']
                    })
            
            if len(plot_data) == 0:
                continue
            
            # Sort by x position
            plot_data = sorted(plot_data, key=lambda x: x['x_pos'])
            x_positions = [d['x_pos'] for d in plot_data]
            y_means = [d['y_mean'] for d in plot_data]
            y_stds = [d['y_std'] for d in plot_data]
            counts = [d['count'] for d in plot_data]
            
            # Plot line with mean values
            color = colors[j]
            style = line_styles[j % len(line_styles)]
            
            ax.plot(x_positions, y_means, 
                   marker='o', 
                   label=f'$p_{{\\text{{miss}}}}$ = {p_miss}', 
                   linewidth=3.5, 
                   markersize=13,
                   color=color,
                   linestyle=style)
            
            # Customize tick parameters
            ax.tick_params(axis='y', labelsize=15, colors='#2F4F4F')
            ax.tick_params(axis='x', labelsize=15, colors='#2F4F4F')

            # Add confidence interval based on standard deviation
            if confidence_interval and len(y_means) > 0:
                ci_lower = [y_mean - y_std for y_mean, y_std in zip(y_means, y_stds)]
                ci_upper = [y_mean + y_std for y_mean, y_std in zip(y_means, y_stds)]
                
                ax.fill_between(x_positions, ci_lower, ci_upper, 
                              alpha=0.15, color=color)
            
            # Mark minimum point with a star
            if len(y_means) > 0:
                min_idx = np.argmin(y_means)
                min_x = x_positions[min_idx]
                min_y = y_means[min_idx]
                
                ax.plot(min_x, min_y, 
                       marker='*', 
                       markersize=20, 
                       color=color,
                       markeredgecolor='red',
                       markeredgewidth=2,
                       zorder=30)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        
        
        # Customize subplot
        ax.set_xlabel(hyper, fontsize=20)
        ax.set_title(f'{dataset}', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks and labels
        valid_positions = []
        valid_labels = []
        for pos, label in enumerate(desired_order):
            # Only show ticks for hyperparameter values that exist in this dataset
            dataset_hyper_values = dataset_data['hyper_display'].unique()
            if label in dataset_hyper_values:
                valid_positions.append(pos)
                valid_labels.append(label)
        
        if valid_positions:
            ax.set_xticks(valid_positions)
            ax.set_xticklabels(valid_labels)
        
        # Add legend to the last subplot
        # if i == len(datasets) - 1:
        #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if there are items to show
        fig.legend(handles, labels, 
                  loc='lower center',           # Position at bottom center
                  bbox_to_anchor=(0.5, -0.5), # Fine-tune position 
                  ncol=len(p_miss_values),      # Arrange horizontally
                  fontsize=16,                  # Large font size
                  frameon=True,                 # Show frame around legend
                  fancybox=True,               # Rounded corners
                  shadow=True)
    # Single y-axis label for all subplots
    # fig.supylabel(metric, fontsize=20, x=0.005)
    # plt.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.supylabel(metric, fontsize=20, x=0.01)
    
    # Adjust subplot spacing - make them closer together
    plt.subplots_adjust(wspace=0.02)  # Reduce horizontal space between subplots
    plt.tight_layout(pad=0.5, w_pad=0.02)  # Also reduce w_pad for tighter spacing
    
    plt.subplots_adjust(
        left=0.05,    # Left margin
        right=0.95,   # Right margin  
        top=0.9,      # Top margin
        bottom=0.1,   # Bottom margin
        wspace=0.16,  # Width spacing between subplots
        hspace=0.2    # Height spacing (if you had multiple rows)
        )
    
    # Save if requested
    if save_pdf:
        fig.savefig(save_pdf, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {save_pdf}")
    
    return fig, axes

