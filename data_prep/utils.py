#!/usr/bin/env python
import re
import click
import polars as pl
import pandas as pd
import anndata as ad
import numpy as np

def normalize_mad(df: pl.DataFrame, meta: pl.DataFrame, batch_col="Metadata_batch_id", control_str="PHB000001") -> pl.DataFrame:
    """
    Performs Z-MAD normalization on a Polars DataFrame.
    
    Parameters:
        df (pl.DataFrame): Feature DataFrame (only numerical values).
        meta (pl.DataFrame): Metadata DataFrame.
        batch_col (str): Column in meta that contains batch information.
        control_str (str): Value in batch_col that defines the control group.
        use_clipping (bool): If True, clips extreme values to 1st and 99th percentiles.

    Returns:
        pl.DataFrame: Normalized feature DataFrame concatenated with metadata.
    """ 
    if batch_col not in meta.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in metadata.")

    # Merge metadata with feature DataFrame on row index
    df = df.with_row_count(name="row_idx")
    meta = meta.with_row_count(name="row_idx")
    df = df.join(meta, on="row_idx", how="inner")

    # Identify numeric columns (excluding metadata)
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

    # Filter only control samples
    df_control = df.filter(pl.col(batch_col) == control_str)
    # Ensure control group is not empty
    if df_control.is_empty():
        raise ValueError(f"No control samples found with '{batch_col}' == '{control_str}'.")

    # Compute median and MAD using only numeric columns
    control_median = df_control.select([pl.col(c).median().alias(c) for c in numeric_cols])
    control_mad = df_control.select([(pl.col(c) - control_median[c]).abs().median().alias(c) for c in numeric_cols])

    # Replace zero MAD values to prevent division by zero
    control_mad = control_mad.with_columns([
        pl.when(pl.col(c) == 0).then(1e-8).otherwise(pl.col(c)).alias(c) for c in control_mad.columns
    ])

    # Apply Z-MAD normalization
    df_norm = df.with_columns([(pl.col(c) - control_median[c]) / control_mad[c] for c in numeric_cols])
    # Drop "row_idx" before concatenating metadata
    df_norm = df_norm.drop("row_idx")
    meta = meta.drop("row_idx")
    return df_norm


def normalize_zscore(df: pl.DataFrame, meta: pl.DataFrame, batch_col="batchid", control_str="PHB000001") -> pl.DataFrame:
    """
    Normalizes features using Z-score normalization.
    Parameters:
        df (pl.DataFrame): Feature DataFrame (only numerical values)
        meta (pl.DataFrame): Metadata DataFrame
        batch_col (str): Column in meta that contains batch information
        control_str (str): Value in batch_col that defines the control group
    Returns:
        pl.DataFrame: Normalized feature DataFrame
    """
    # Filter only control samples
    df_control = df.filter(meta[batch_col] == control_str)
    
    # Compute mean and std deviation
    control_mean = df_control.mean()
    control_std = df_control.std()
    
    # Replace zeros in std to prevent division by zero
    control_std = control_std.with_columns([
        pl.when(pl.col(c) == 0).then(1e-8).otherwise(pl.col(c)).alias(c) for c in control_std.columns
    ])
    
    # Normalize
    df_norm = (df - control_mean) / control_std
    df_out = pl.concat([df_norm, meta])
    return df_out

def normalize_standard(df: pl.DataFrame, meta: pl.DataFrame, batch_col="batchid", control_str="PHB000001") -> pl.DataFrame:
    """
    Standard normalization: Removes low-variance features and applies Z-score normalization.
    Parameters:
        df (pl.DataFrame): Feature DataFrame (only numerical values)
        meta (pl.DataFrame): Metadata DataFrame
        batch_col (str): Column in meta that contains batch information
        control_str (str): Value in batch_col that defines the control group
    Returns:
        pl.DataFrame: Normalized feature DataFrame
    """
    # Remove low-variance features
    std_dev = df.std()
    df = df.select([col for col in df.columns if std_dev[col].item() > 0.001])

    # Filter only control samples
    df_control = df.filter(meta[batch_col] == control_str)
    
    # Compute mean and std deviation
    control_mean = df_control.mean()
    control_std = df_control.std()
    
    # Normalize
    df_norm = (df - control_mean) / control_std
    df_out = pl.concat([df_norm, meta])
    return df_out

def aggregate(df: pl.DataFrame, meta_cols: list, feat_cols: list, strategy="median") -> pl.DataFrame:
    """
    Aggregates feature columns by grouping on the metadata columns.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing both metadata and feature columns.
        meta_cols (list): List of columns to group by.
        feat_cols (list): List of feature columns to aggregate.
        strategy (str): Aggregation strategy: "median" or "mean".
        
    Returns:
        pl.DataFrame: Aggregated DataFrame.
    """
    if strategy.lower() == "median":
        aggs = [pl.col(col).median().alias(col) for col in feat_cols]
    elif strategy.lower() == "mean":
        aggs = [pl.col(col).mean().alias(col) for col in feat_cols]
    else:
        raise ValueError(f"Strategy '{strategy}' is not defined. Supported strategies: 'median', 'mean'.")
    
    return df.group_by(meta_cols).agg(aggs)


def get_metadata_polars(df: pl.DataFrame):
    """
    Splits a Polars DataFrame into features and metadata.
    Any column whose name contains "Metadata" is treated as metadata.
    
    Returns:
        feat (pl.DataFrame): DataFrame of feature columns.
        meta (pl.DataFrame): DataFrame of metadata columns.
    """
    meta_cols = [col for col in df.columns if "Metadata" in col]
    feat_cols = [col for col in df.columns if "Metadata" not in col]
    feat = df.select(feat_cols)
    meta = df.select(meta_cols)
    return feat, meta


def ensure_metadata_plate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Checks if 'Metadata_Plate' exists in the DataFrame columns.
    If not, but 'Metadata_Barcode' exists, creates 'Metadata_Plate'
    with the same values as 'Metadata_Barcode'.
    
    Returns:
        The updated DataFrame.
    """
    if "Metadata_Plate" not in df.columns:
        if "Metadata_Barcode" in df.columns:
            df = df.with_column(pl.col("Metadata_Barcode").alias("Metadata_Plate"))
    return df

# Data split



# feat select
