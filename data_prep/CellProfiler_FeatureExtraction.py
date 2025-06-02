import polars as pl
from sqlalchemy import create_engine
import re
import gc
import numpy as np
import pandas as pd
import os
import sys
import time
#from sqlalchemy import create_engine
import re
import matplotlib.pyplot as plt
import gc
#from pycytominer.operations.correlation_threshold import correlation_threshold
import polars as pl
import pandas as pd
# import plotly.figure_factory as ff
# import plotly.subplots as sp
# import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
from typing import Union, Literal, Tuple, Set, List, Dict
# from cytominer_eval import evaluate
import scanpy as sc
import anndata as ad
db_uri = #confidential

class CellProfilerData:
    def __init__(self, project, analysis_id_feat, acq_id,barcode,z ):
        self.project = project
        self.analysis_id_feat = analysis_id_feat
        self.acq_id = acq_id
        self.df = None
        self.df_qc = None
        self.df_meta = None
        self.dataframes = None
        self.locations = None
        self.barcode = barcode
        self.z =z
    def get_data_from_server(self,type='feature'):
        if type == 'feature':
            type = 'cp-features'
        elif type == 'qc':
            type = 'cp-qc'
        else:
            raise ValueError("Type must be either 'feature' or 'qc'")
        query = f"""
            SELECT *
            FROM image_analyses_per_plate
        WHERE project LIKE '{self.project}%%'
        AND meta->>'type' = '{type}'
        AND analysis_date IS NOT NULL
        ORDER BY plate_barcode 
        """
        engine = create_engine(db_uri)
        connection = engine.connect()
        self.df = pl.read_database(query, connection)
        connection.close()
        return self.df

    def check_duplicate_analysis(self):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call get_data_from_server() first.")

        df = self.df.filter(pl.col('plate_acq_id') == self.acq_id)
        self.df = df
        print(df)
        df_with_count = df.group_by('plate_acq_id').agg(pl.count().alias('size'))
        print(df.select('plate_acq_id'))
        df_dupes = df_with_count.filter(pl.col('size') > 1)
        if df_dupes.is_empty():
            print("OK, no duplicate results found")
        else:
            #print("WARNING! Duplicate results found")
            #print(self.df.select('analysis_id'))
            print(self.analysis_id_feat)
            self.df = self.df.filter(pl.col('analysis_id') == self.analysis_id_feat)
            
            print(self.df)
            print(self.df.select('analysis_id'))


    def load_feature_data(self,featureFileNames=['featICF_nuclei', 'featICF_cells', 'featICF_cytoplasm']):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call get_data_from_server() first.")
        print(self.df.select('results'))
        #featureFileNames = ['featICF_nuclei', 'featICF_cells', 'featICF_cytoplasm']
        print(self.df.select('results').to_series())
        dataframes = {}
        for i in featureFileNames:
            base = self.df.select('results').to_series()[0]
            file = base + i + '.parquet'
            print(file)
            dataframe = pl.read_parquet(file)
            new_columns = [str(col) + '_' + re.sub('_.*', '', re.sub('featICF_', '', i)) for col in dataframe.columns]
            dataframe = dataframe.rename({old: new for old, new in zip(dataframe.columns, new_columns)})
            dataframes[i] = dataframe
        self.dataframes = dataframes
    

    def merge_dataframes(self):
        if self.dataframes is None:
            raise ValueError("Feature data is not loaded. Call load_feature_data() first.")
        
        df = self.dataframes['featICF_nuclei']
        print('size', df.shape)
        
        df = df.join(
            self.dataframes['featICF_cells'],
            left_on=['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei', 'Parent_cells_nuclei'],
            right_on=['Metadata_Barcode_cells', 'Metadata_Site_cells', 'Metadata_Well_cells', 'ObjectNumber_cells'],
            how='left'
        )
        print('size', df.shape)
        df = df.join(
            self.dataframes['featICF_cytoplasm'],
            left_on=['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei', 'Parent_cells_nuclei'],
            right_on=['Metadata_Barcode_cytoplasm', 'Metadata_Site_cytoplasm', 'Metadata_Well_cytoplasm', 'ObjectNumber_cytoplasm'],
            how='left'
        )
        print('size', df.shape)
        df = df.drop_nulls()
        print('size', df.shape)
        del self.dataframes
        gc.collect()
        df = df.with_columns([
            pl.lit(self.z).alias('Z')
        ])
        self.df = df


    def split_feat_metadata(self):
        def is_meta_column(c):
            for ex in '''
                ^[a-z]
                Metadata
                ^Count
                ImageNumber
                Object
                Parent
                Children
                Plate
                Well
                Location
                _[XYZ]_
                _[XYZ]$
                BoundingBox
                Phase
                Orientation
                Angle
                Scale
                Scaling
                Width
                Height
                Group
                FileName
                PathName
                URL
                Execution
                ModuleError
                LargeBrightArtefact
                MD5Digest
                '''.split():
                if re.search(ex, c):
                    return True
            return False

        if self.df is None:
            raise ValueError("DataFrame is not merged. Call merge_dataframes() first.")
        self.df = self.df.rename({
        'Metadata_Barcode_nuclei': 'Metadata_Plate',
        'Metadata_Well_nuclei': 'Metadata_Well',
        'Metadata_Site_nuclei': 'Metadata_Site'
        })
        self.meta = self.df.select('Metadata_Plate', 'Metadata_Well', 'Metadata_Site','ObjectNumber_nuclei').with_row_index()
        self.locations = self.df
        self.meta = self.meta.with_columns([
            pl.lit(self.z).alias('Z')
        ])
        feat = self.df.select([col for col in self.df.columns
                              if '[Float64]' in str(self.df.select(col).dtypes) or '[Float32]' in str(self.df.select(col).dtypes) or 'int' in str(self.df[col].dtype)
                              if not is_meta_column(col)])
        blocklist_features = (
        [col for col in feat.columns if "Granularity_12" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Correlation_Manders" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Correlation_Manders" in col and "_cytoplasm" in col] +
        [col for col in feat.columns if "Correlation_RWC" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Granularity_14" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Granularity_15" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Granularity_16" in col and "_nucleus" in col] +
        [col for col in feat.columns if "Granularity_3" in col and "_cytoplasm" in col] +
        [col for col in feat.columns if "Granularity_5" in col and "_cytoplasm" in col] +
        [col for col in feat.columns if "Granularity_4" in col and "_cytoplasm" in col]
        )                
        feat = feat.drop(blocklist_features)
        #self.df = self.df.select(feat.columns,'Metadata_Well')
        self.df = self.df.select(feat.columns).with_row_index()
        
    def load_compound(self,barcode,project,csv_file=False):
        if csv_file:
            compound = pl.read_csv(csv_file)
            compound = compound.filter(pl.col('barcode') == barcode )
            print(compound)
            # rename colum batch_id to batch_nr
            #compound = compound.rename({'batchid': 'batch_id'})
            #compound = compound.rename({'well-id': 'well_id'})
        else:
            db_uri = #confidential

            query = f"""
                    SELECT *
                    FROM plate_v1
                    WHERE layout_id LIKE '%%{project}%%'
                    """
            compound = pd.read_sql_query(query, db_uri)
            compound = compound[compound['barcode'] == barcode]
            # Query database and store result in pandas dataframe
            print("Select table with database...please wait")
            #drop duplicates
            compound['batch_id'] = compound['batch_id'].replace('AZ000001', 'PHB000001')
            compound = compound.drop_duplicates(subset=['well_id'])
            compound = pl.DataFrame(compound)
            print(compound)
        #merge with metadata
        self.meta = self.meta.join(compound.select('well_id','batch_id','cmpd_conc'), left_on='Metadata_Well',right_on='well_id', how='left')
        return compound
        


    def aggregate_dataframe_site(self, function='median'):
        if self.df is None:
            raise ValueError("DataFrame is not merged. Call merge_dataframes() first.")
        
        if function == 'median':
            df = self.df.group_by(['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']).agg(
        
                [pl.col(col).median().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']]
            )
        elif function == 'mean':
            df = self.df.group_by(['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']).agg(
                [pl.col(col).mean().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nuclei', 'Metadata_Site_nuclei', 'Metadata_Well_nuclei']]
            )
        self.df = df

    def aggregate_dataframe_well(self, function='median'):
        if self.df is None:
            raise ValueError("DataFrame is not merged. Call merge_dataframes() first.")
        
        if function == 'median':
            df = self.df.group_by(['Metadata_Barcode_nuclei', 'Metadata_Well_nuclei']).agg(
        
                [pl.col(col).median().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nuclei', 'Metadata_Well_nuclei']]
            )
        elif function == 'mean':
            df = self.df.group_by(['Metadata_Barcode_nuclei', 'Metadata_Well_nuclei']).agg(
                [pl.col(col).mean().alias(col) for col in self.df.columns if col not in ['Metadata_Barcode_nuclei', 'Metadata_Well_nuclei']]
            )
        self.df = df
    #blocklist_features = [col for col in normalized_profiles.columns if "Correlation_Manders" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Correlation_RWC" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Granularity_14" in col and "_nuclei" in col] + [col for col in normalized_profiles.columns if "Granularity_15" in col and "_nuclei" in col] +[col for col in normalized_profiles.columns if "Granularity_16" in col and "_nuclei" in col]
    #features = [feat for feat in normalized_profiles_merge.columns if feat not in meta_features and feat not in blocklist_features and feat not in meta_df_features]

    def Zmean(self):
        if 'batch_id' in self.meta.columns:
            df_DMSO_meta = self.meta.filter(pl.col('batch_id')== 'PHB000001')
        else:
            df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')

        float_columns = [col for col in df_DMSO_feat.columns if col != 'index']
        #df_DMSO = self.df.select(float_columns)
        df_DMSO = df_DMSO_feat.select(float_columns)

        mu = df_DMSO.select(float_columns).mean()
        std = df_DMSO.select(float_columns).std()
        #df = self.df
        for col in mu.columns:
            if mu[col].is_null().any():
                raise RuntimeError(f"some mean value in column {col} is nan?!")
            if mu[col].is_infinite().any():
                raise RuntimeError(f"some mean value in column {col} is infinite?!")
                   ### OLD PYTHON
        #std = std.select([pl.col(c).map_dict({0: 1}, default=pl.col(c)) for c in std.columns])
        #self.df = self.df.with_columns([(pl.col(c) - mu[c]) / (std[c]+0.01) for c in mu.columns])
        std = std.select([
        pl.when(pl.col(c) == 0).then(1).otherwise(pl.col(c)).alias(c) for c in std.columns
        ])
        self.df = self.df.with_columns([(pl.col(c) - mu[c]) / (std[c]) for c in mu.columns])
        # make all columns float 32
        for col in self.df.columns:
            self.df = self.df.with_columns(pl.col(col).cast(pl.Float32))
        
        # replace 0 with 1 (specifically not clip) to avoid div by zero
    def Zmad(self,use_clipping=True):
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call merge_dataframes() first.")
        if self.meta is None:
            raise ValueError("Metadata DataFrame is not loaded.")
        # if use_clipping:
        #     lower_quantile = self.df.quantile(0.01)
        #     upper_quantile = self.df.quantile(0.99)
        #     print("calced quantiles")

        #     for col in self.df.columns:
        #         if col != 'index': 
        #             self.df = self.df.with_columns(pl.col(col).clip(lower=lower_quantile[col],upper=upper_quantile[col]))
        
        df_DMSO_meta = self.meta.filter(pl.col('batch_id') == 'PHB000001')
        df_DMSO_index = df_DMSO_meta.select('index')
        df_DMSO_feat = self.df.join(df_DMSO_index, on='index', how='inner')

        float_columns = [col for col in df_DMSO_feat.columns if col != 'index']
        df_DMSO = self.df.select(float_columns)
        median = df_DMSO.select(float_columns).median()
        mad = df_DMSO.select([(pl.col(c) - pl.col(c).median()).abs().alias(c) for c in float_columns])
        mad = pl.concat([self.df.select((pl.col(c)-pl.col(c).median()).abs().median()) for c in self.df.select(float_columns).columns], how='horizontal')
        #mad = mad.select([pl.col(c).map_dict({0: 1e-8}, default=pl.col(c)) for c in mad.columns])
        df_standardized = self.df.with_columns([(pl.col(c) - median[c]) / (mad[c]) for c in median.columns])
        
        # remove columns with all zeros
        
        # Check for null or infinite medians and raise errors if found
        
        #df_standardized = df_standardized.select([c for c in df_standardized.columns if df_standardized[c].sum() != 0])
        self.df = df_standardized
    def normalize_mad(self, batch_col="batch_id", control_str="PHB000001") -> pl.DataFrame:
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
        if batch_col not in self.meta.columns:
            raise ValueError(f"Batch column '{batch_col}' not found in metadata.")

        # Merge metadata with feature DataFrame on row index
        df = self.df.with_row_count(name="row_idx")
        meta = self.meta.with_row_count(name="row_idx")
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
        self.df = df_norm.drop("row_idx")
        meta = meta.drop("row_idx")

def drop_corr(df, threshold=0.9, samples='all'):
            from pycytominer.operations.correlation_threshold import correlation_threshold
            exclude = correlation_threshold(
                df,
                features=list(df.columns),
                threshold=threshold,
                samples=samples,
            )
            print(f'Dropping {len(exclude)=} columns (starting from {df.shape=})')
            #return df.map_data(lambda df: df.drop(columns=exclude))
            return df.drop(exclude, axis=1)
def drop_constant_polars(df: pl.DataFrame, columns, threshold: float = 0.001) -> pl.DataFrame:
    drop_columns = []
    for column in tqdm(columns):
        value_counts = df.group_by(column).agg(pl.count().alias('counts')).sort(by='counts', descending=True)
        most_common_count = value_counts['counts'][0]
        total_count = df.height
        fraction = most_common_count / total_count
        if fraction > threshold:
            drop_columns.append(column)
    print(f"{len(drop_columns)} columns to drop due to constant values")
    return df.drop(drop_columns), drop_columns

# Function to drop low-variance columns in a Polars DataFrame
def drop_low_variance_pl(df, columns_to_check, threshold: float = 0.01):
    valid_columns = [col for col in columns_to_check if col in df.columns]
    columns_to_drop = []
    for col in valid_columns:
        variance = df.select(pl.col(col).var().alias("variance")).to_pandas().iloc[0, 0]
        if variance < threshold:
            columns_to_drop.append(col)
    df = df.drop(columns_to_drop)
    if columns_to_drop:
        print(f"Dropped {len(columns_to_drop)} columns for low variance")
    else:
        print("No columns dropped due to low variance.")
    return df


def merge_df_qc (df,df_qc):
    #display(df_qc)
    #display(df)
    df_merged = pd.merge(df, df_qc, left_on=['Metadata_Well_nuclei','Metadata_Site_nuclei','Z'],right_on=['Metadata_Well','Metadata_Site','Z'], how='left',suffixes=('', '_qc'))    
    #locations = pd.merge(locations, df_qc, left_on=['Metadata_Well','Metadata_Site','Z'],right_on=['Metadata_Well','Metadata_Site','Z'], how='left',suffixes=('', '_qc'))
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    #locations = locations.loc[:,~locations.columns.duplicated()]
    df_remove_flagged = df_merged[df_merged['Total'] == 0 ]
    #locations = locations[locations['Total'] == 0]
    print("Reduction by", (len(df_merged))-(len(df_remove_flagged)) )
    print("Number of flagged instances in QC was", len(df_qc[df_qc['Total'] == 1]))
    #columns_to_drop = ['Outlier', 'Total']
    #feature_columns = [fc for fc in df_remove_flagged.columns if all(exclude not in fc for exclude in columns_to_drop)]
    #df_remove_flagged = df_remove_flagged[feature_columns]
    df = pl.from_pandas(df_remove_flagged.drop(columns=df_qc.columns))
    df_qc = pl.from_pandas(df_remove_flagged.drop(columns=df.columns))
    #df_indices = df.select(pl.col("index")) 
    #locations = locations.filter(pl.col("index").is_in(df_indices))

    # display(df)
    #display(df_qc)
    return df