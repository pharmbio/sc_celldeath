import scanpy as sc 
import anndata as ad
import polars as pl
import os
import sys
import matplotlib.pyplot as plt
from scperturb import *
import seaborn as sns
import click  # Import the click library
import re
import numpy as np 
import pandas as pd

PROJECT_DIR = "/home/jovyan/share/data/analyses/benjamin/Single_cell_project_rapids/Beactica"

@click.command()
@click.option('--input', '-i', 'input_dir', required=True, help='Input directory path merged with project root.')
@click.option('--output', '-o', 'output_file', required=False, help='Output file name (or directory).')
@click.option('--compound_col', '-c', 'compound_col', required=True, help='Name of the compound column for calculations.')
@click.option('--sample_size', '-s', 'sample_size', required=True, type = int, help='Group sample size.')
@click.option('--num_perms', '-p', 'permut_num', required=True, type = int, help='Number of perturbations.')



def basic_analysis(input_dir, output_file, compound_col, sample_size, permut_num):
    # Merge the input directory with the project root
    
    adata = ad.read(input_dir)
    print("Data imported")
    anndata_group = equal_subsampling(adata, compound_col, N_min= sample_size)

    print("Running etest!")
    df = etest(anndata_group, obs_key=compound_col, obsm_key='X_pca', dist='sqeuclidean', control='[DMSO]', alpha=0.05, runs=permut_num, n_jobs=-1)
    print(df)
    print("Saving test plots.")
    
    df.loc[df.index=='[DMSO]', 'significant_adj'] = '[DMSO]'
    df['neglog10_pvalue_adj'] = -np.log10(df['pvalue_adj'])
    with sns.axes_style('whitegrid'):
        sns.scatterplot(data=df, y='neglog10_pvalue_adj', x='edist', hue='significant_adj', palette={True: 'tab:green', False: 'tab:red', '[DMSO]': 'tab:orange'}, s=30)
    plt.title('E-test results')
    plt.xlabel('E-distance from control')
    plt.ylabel('E-test neg log10 of adjusted p-value')
    plt.savefig(f"etest_res_{sample_size}_samples_{permut_num}_perms_grit_all.png")
    plt.show()
    df2 = pd.DataFrame(df)
    #bool_map = {
    #'[DMSO]': False,
    # Add other mappings as necessary
    #}
    #df2['significant_adj'] = df2['significant_adj'].map(bool_map).astype(bool)
 
    df.to_csv(f"etest_res_specs5k_{sample_size}_samples_{permut_num}_perms.csv")

if __name__ == "__main__":
    basic_analysis()