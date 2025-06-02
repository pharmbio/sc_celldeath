import polars as pl
import numpy as np
import pandas as pd
import os
import sys
import time
from sqlalchemy import create_engine
import re
import matplotlib.pyplot as plt
import gc
import polars as pl
import pandas as pd

import numpy as np
from collections import defaultdict
from typing import Union, Literal, Tuple, Set, List, Dict
from sklearn.preprocessing import StandardScaler

db_uri = #confidential
class CellProfilerQC:
    def __init__(self, project, analysis_id_qc, acq_id,z):
        self.project = project
        self.analysis_id_qc = analysis_id_qc
        self.acq_id = acq_id
        self.df = None
        self.df_qc = None
        self.df_meta = None
        self.dataframes = None
        self.dataframes_list = None
        self.z = z

    def get_data_from_server(self):
        query = f"""
        SELECT *
        FROM image_analyses_per_plate
        WHERE project LIKE '{self.project}%%'
        AND meta->>'type' = 'cp-qc'
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

        self.df = self.df.filter(pl.col('plate_acq_id') == self.acq_id)
        df_with_count = self.df.group_by('plate_acq_id').agg(pl.count().alias('size'))
        #print(self.df.select('plate_acq_id'))
        df_dupes = df_with_count.filter(pl.col('size') > 1)
        if df_dupes.is_empty():
            print("OK, no duplicate results found")
            print(self.df.select('analysis_id'))

        else:
            #print("WARNING! Duplicate results found")
            #print(self.df.select('analysis_id'))
            self.df = self.df.filter(pl.col('analysis_id') == self.analysis_id_qc)
            #print(self.df.select('analysis_id'))
    def load_file(self):
        print(self.df.select('results'))
        print(self.df.select('plate_barcode'))
        if self.project == 'pelago':
            file = self.df.select('results') + 'QC_image_'+'P102787-HepG2-2-48h-P01-L2-MoA'+'.parquet'
        else:
            file = self.df.select('results') + 'qcRAW_images_'+self.df.select('plate_barcode') +'.parquet'
        #display(file['results'])
        self.df = pl.read_parquet(file['results'][0])
        self.df= self.df.with_columns([
            pl.lit(self.z).alias('Z')
        ])
       
    def select_measures(self):
        def remove_not_useful_measure(list_measures):
            NotSoUseful = ['TotalArea', 'Scaling', 'TotalIntensity', 'Correlation', 'PercentMinimal'
                ,'LocalFocusScore', 'MinIntensity', 'MedianIntensity', 'MADIntensity'
                ,'ThresholdMoG'
                ,'ThresholdBackground'
                ,'ThresholdKapur'
                ,'ThresholdMCT'
                ,'ThresholdOtsu'
                ,'ThresholdRidlerCalvard'
                ,'ThresholdRobustBackground'
                , 'PercentMaximal'
                ]
            for u in NotSoUseful:
                list_measures.remove(u)
        ImageQuality = [s for s in self.df.columns if "ImageQuality_" in s]
        ImageQualityModule = [s.replace('ImageQuality_', '') for s in ImageQuality]
        ImageQualityMeasures = sorted(list(set([re.sub('_.*', '', s) for s in ImageQualityModule])))
        CountMeasures = len(ImageQualityMeasures)
        print('ImageQuality module has measured '
            + str(CountMeasures) + ' parameters: ' + ', '.join(ImageQualityMeasures))
        remove_not_useful_measure(ImageQualityMeasures)
        self.dataframes = {}
        for z in ImageQualityMeasures:
            parameter = [s for s in ImageQuality if ('_' + z) in s]
            self.dataframes[z]=self.df[parameter].to_pandas()
        self.dataframes_list = sorted(list(self.dataframes.keys()))
        ChannelNames = []
        for c in list(self.dataframes[self.dataframes_list[0]].columns):
            ChannelNames.append(re.sub('.*_', '', c))
        return ImageQualityMeasures,ChannelNames


def create_flags_automatically(DataFrameList,DataFrameDictionary,data,ChannelNames,threshold,count_nuclei):
    LowerLimitScaled = -threshold #float('-inf')
    UpperLimitScaled = threshold #float('inf')
    Flags = []
    #ChannelNames = []
    #for c in list(DataFrameDictionary[DataFrameList[0]].columns):
        #ChannelNames.append(re.sub('.*_', '', c))

    for p in range(0, len(DataFrameList)):
        CurrentDataFrame = DataFrameDictionary.get(DataFrameList[p])
        
        x_unscaled = CurrentDataFrame.values 
        x_scaled = StandardScaler().fit_transform(x_unscaled)
        CurrentDataFrameScaled = pd.DataFrame(x_scaled, columns = ChannelNames)

        NewFlagSc = 'OutlierScaled' + '_' + DataFrameList[p] + '_' + str(LowerLimitScaled) + '_' + str (UpperLimitScaled)
        Flags.append(NewFlagSc)
        data[NewFlagSc] = 0

        CurrentDataFrameOutliersMetadata = data[(CurrentDataFrameScaled.values >= UpperLimitScaled).
                                        any(1) | (CurrentDataFrameScaled.values <= LowerLimitScaled).
                                        any(1)][['Metadata_Barcode', 'Metadata_Well', 'Metadata_Site','Z']]
        CurrentDataFrameOutliersValues = CurrentDataFrameScaled[(CurrentDataFrameScaled.values >= UpperLimitScaled).
                                        any(1) | (CurrentDataFrameScaled.values <= LowerLimitScaled).
                                        any(1)]
        CurrentDataFrameScaledOutliers = CurrentDataFrameOutliersMetadata.merge(CurrentDataFrameOutliersValues,
                                                                        left_index=True, right_index=True)

        Outliers = CurrentDataFrameScaledOutliers.index.values.tolist()
        data.loc[Outliers,NewFlagSc] = 1
    data['Total'] = data[Flags].max(axis = 1)
    data.loc[data['Count_nuclei'] < count_nuclei, 'Total'] = 1
    #data['Total'] = 1data['Count_nuclei']
    # data ['Total'] = 1 if Count_nuclei >3
    #data['Total'] = data['Total'] + data
    Flags.append('Total')
    print(data[Flags].sum())
    df_flags = data[['Metadata_Barcode', 'Metadata_AcqID', 'Metadata_Well', 'Metadata_Site', 'Count_nuclei','Z'] + list(DataFrameDictionary[DataFrameList[0]].columns) + Flags]
    
    return df_flags
