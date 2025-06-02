import re
import click
import polars as pl
import pandas as pd
import anndata as ad
import numpy as np
from utils import *
import os
#############################################
# Processing for one plate
#############################################

def process_plate(df_plate: pl.DataFrame, feature_extractor: str, norm_method: str, control_str: str) -> pl.DataFrame:
    """
    Processes a single plate's data by first aggregating feature data by metadata, 
    then applying normalization based on the control samples.

    Returns:
        pl.DataFrame: Aggregated and normalized Polars DataFrame.
    """
    if feature_extractor in ['DINO', 'DeepProfiler', 'CellProfiler']:
        feat, meta = get_metadata_polars(df_plate)
    else:
        raise ValueError("Unknown feature_extractor specified.")

    # Step 1: Aggregate first (group by all metadata columns starting with "Metadata")
    #group_cols = [col for col in meta.columns if col.startswith("Metadata") if "Site" not in col]

    """Include SIte in grouping"""
    group_cols = [col for col in meta.columns if col.startswith("Metadata") and "Site" not in col]
    print(group_cols)
    combined = feat.hstack(meta)
    agg_df = aggregate(combined, meta_cols=group_cols, feat_cols=feat.columns, strategy="median")

    # Step 2: Normalize the aggregated features if "Metadata_batch_id" exists
    if "Metadata_batch_id" in agg_df.columns:
        feat = agg_df.select([col for col in agg_df.columns if col not in group_cols])  # Select only numeric feature columns
        meta = agg_df.select(group_cols)  # Select only metadata columns

        if norm_method == "MAD":
            norm_df = normalize_mad(feat, meta, control_str=control_str, batch_col="Metadata_batch_id")
        elif norm_method == "zscore":
            norm_df = normalize_zscore(feat, meta, control_str=control_str, batch_col="Metadata_batch_id")
        elif norm_method == "standard":
            norm_df = normalize_standard(feat, meta, control_str=control_str, batch_col="Metadata_batch_id")
        elif norm_method == "no":
            norm_df = pl.concat([feat, meta], how = "horizontal")
            
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

        # Combine back features and metadata
        feat = norm_df.select(feat.columns)
        meta = norm_df.select(meta.columns)
        final_df = feat.hstack(meta)
        print(final_df.shape)
    else:
        final_df = agg_df  # If no normalization is done, return the aggregated data

    return final_df

def load_input_as_polars(path: str) -> pl.DataFrame:
    """
    Loads an input file as a Polars DataFrame.
    If the file extension is .csv, it uses pl.read_csv().
    If the file extension is .h5ad, it loads the file with anndata,
    converts the AnnData to a pandas DataFrame by combining adata.obs and adata.X,
    and then converts it to a Polars DataFrame.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pl.read_parquet(path)
    elif ext == ".h5ad":
        adata = ad.read_h5ad(path)
        # Convert X to dense if necessary.
        if hasattr(adata.X, "toarray"):
            X = adata.X.toarray()
        else:
            X = adata.X
        # Create a DataFrame from features using adata.var['feature_names']
        feat_df = pd.DataFrame(X, columns=adata.var["feature_names"], index=adata.obs.index)
        # Combine with obs metadata.
        df_pandas = pd.concat([adata.obs, feat_df], axis=1)
        df = pl.from_pandas(df_pandas)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df

#############################################
# Main pipeline using Polars (multiple inputs)
#############################################

@click.command()
@click.option('--input', '-i', 'input', required=True, multiple=True,
              help='Input CSV file(s) containing the data. Provide one or more paths.')
@click.option('--output', '-o', 'output', required=True,
              help='Output AnnData file (H5AD format).')
@click.option('--feature_extractor', '-f', required=True,
              type=click.Choice(['DINO', 'DeepProfiler', 'CellProfiler']),
              help='Feature extractor used. For non-CellProfiler methods, metadata columns are those containing "Metadata". For "CellProfiler", metadata is determined via custom logic.')
@click.option('--norm_method', default="MAD", show_default=True,
              help='Normalization method to apply (e.g., "MAD" or "z").')
@click.option('--control_str', default="control", show_default=True,
              help='Control string used for normalization (must be present in metadata as batch-id).')

def main(input, output, feature_extractor, norm_method, control_str):
    """
    Processes a list of input files (CSV or H5AD), executing normalization and aggregation for each plate individually.
    The pipeline splits each file into features and metadata (metadata columns are identified by containing "Metadata"),
    assigns "Metadata_Barcode" to "Metadata_Plate" if missing, groups data by plate (using "Metadata_Plate" if available),
    processes each plate, and concatenates the results.
    The final output is saved as an AnnData file with features in .X and metadata in .obs.
    """
    results = []  # to store aggregated results from all plates
    input = list(input)
    # Process each input file
    for path in input:
        print(path)
        df = load_input_as_polars(path)
        # Ensure we have a "Metadata_Plate" column.
        df = ensure_metadata_plate(df)
        
        # Process per plate if "Metadata_Plate" is available.
        if "Metadata_Plate" in df.columns:
            plates = df.select("Metadata_Plate").unique().to_series().to_list()
            print("Process plate:",plates)
            for plate in plates:
                df_plate = df.filter(pl.col("Metadata_Plate") == plate)
                if feature_extractor == "DeepProfiler":
                    meta_DP = #read in according metadata path
                    df_plate_merge = df_plate.join(meta_DP.select(["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_cmpdName", "Metadata_cmpdConc"]).unique(), on =["Metadata_Plate", "Metadata_Well", "Metadata_Site"])
                    df_plate_merge = df_plate_merge.with_columns(pl.col("Metadata_cmpdName").alias("Metadata_batch_id"))
                    agg_plate = process_plate(df_plate_merge, feature_extractor, norm_method, control_str)
                else:
                    agg_plate = process_plate(df_plate, feature_extractor, norm_method, control_str)
                results.append(agg_plate)
        else:
            agg_plate = process_plate(df, feature_extractor, norm_method, control_str)
            results.append(agg_plate)
    
    if results:
        final_df = pl.concat(results)
    else:
        raise ValueError("No data processed from input files.")

    # Convert final Polars DataFrame to pandas for AnnData creation.
    final_pd = final_df.to_pandas()
    
    # Determine feature columns as those that are not metadata.
    meta_cols = [col for col in final_pd.columns if "Metadata" in col]
    feat_cols = [col for col in final_pd.columns if col not in meta_cols]
    
    # Create AnnData: features go into .X; metadata into .obs.
    adata = ad.AnnData(X=final_pd[feat_cols].values)
    adata.obs = final_pd[meta_cols].copy()
    adata.var["feature_names"] = feat_cols

    adata.write(output)
    click.echo(f"Saved processed AnnData to {output}")

if __name__ == '__main__':
    main(standalone_mode=False)
