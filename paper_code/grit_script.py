from typing import List, Union
import pandas.api.types as ptypes
import pandas as pd
import numpy as np
import polars as pl
import tqdm
import time
import pyarrow
import numpy as np 
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

def evaluate(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
    grit_control_perts: List[str] = ["None"],
    grit_replicate_summary_method: str = "mean"
):
    r"""Evaluate profile quality and strength.

    For a given profile dataframe containing both metadata and feature measurement
    columns, use this function to calculate profile quality metrics. The function
    contains all the necessary arguments for specific evaluation operations.

    Parameters
    ----------
    profiles : pandas.DataFrame
        profiles must be a pandas DataFrame with profile samples as rows and profile
        features as columns. The columns should contain both metadata and feature
        measurements.
    features : list
        A list of strings corresponding to feature measurement column names in the
        `profiles` DataFrame. All features listed must be found in `profiles`.
    meta_features : list
        A list of strings corresponding to metadata column names in the `profiles`
        DataFrame. All features listed must be found in `profiles`.
    replicate_groups : {str, list, dict}
        An important variable indicating which metadata columns denote replicate
        information. All metric operations require replicate profiles.
        `replicate_groups` indicates a str or list of columns to use. For
        `operation="grit"`, `replicate_groups` is a dict with two keys: "profile_col"
        and "replicate_group_col". "profile_col" is the column name that stores
        identifiers for each profile (can be unique), while "replicate_group_col" is the
        column name indicating a higher order replicate information. E.g.
        "replicate_group_col" can be a gene column in a CRISPR experiment with multiple
        guides targeting the same genes. See also
        :py:func:`cytominer_eval.operations.grit` and
        :py:func:`cytominer_eval.transform.util.check_replicate_groups`.
    operation : {'replicate_reproducibility', 'precision_recall', 'grit', 'mp_value'}, optional
        The specific evaluation metric to calculate. The default is
        "replicate_reproducibility".
    groupby_columns : List of str
        Only used for operation = 'precision_recall' and 'hitk'
        Column by which the similarity matrix is grouped and by which the operation is calculated.
        For example, if groupby_column = "Metadata_broad_sample" then precision/recall is calculated for each sample.
        Note that it makes sense for these columns to be unique or to span a unique space
        since precision and hitk may otherwise stop making sense.
    similarity_metric: {'pearson', 'spearman', 'kendall'}, optional
        How to calculate pairwise similarity. Defaults to "pearson". We use the input
        in pandas.DataFrame.cor(). The default is "pearson".

    Returns
    -------
    float, pd.DataFrame
        The resulting evaluation metric. The return is either a single value or a pandas
        DataFrame summarizing the metric as specified in `operation`.

    Other Parameters
    -----------------------------
    replicate_reproducibility_quantile : {0.95, ...}, optional
        Only used when `operation='replicate_reproducibility'`. This indicates the
        percentile of the non-replicate pairwise similarity to consider a reproducible
        phenotype. Defaults to 0.95.
    replicate_reproducibility_return_median_cor : bool, optional
        Only used when `operation='replicate_reproducibility'`. If True, then also
        return pairwise correlations as defined by replicate_groups and
        similarity metric
    precision_recall_k : int or list of ints {10, ...}, optional
        Only used when `operation='precision_recall'`. Used to calculate precision and
        recall considering the top k profiles according to pairwise similarity.
    grit_control_perts : {None, ...}, optional
        Only used when `operation='grit'`. Specific profile identifiers used as a
        reference when calculating grit. The list entries must be found in the
        `replicate_groups[replicate_id]` column.
    grit_replicate_summary_method : {"mean", "median"}, optional
        Only used when `operation='grit'`. Defines how the replicate z scores are
        summarized. see
        :py:func:`cytominer_eval.operations.util.calculate_grit`
    mp_value_params : {{}, ...}, optional
        Only used when `operation='mp_value'`. A key, item pair of optional parameters
        for calculating mp value. See also
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`
    enrichment_percentile : float or list of floats, optional
        Only used when `operation='enrichment'`. Determines the percentage of top connections
        used for the enrichment calculation.
    hitk_percent_list : list or "all"
        Only used when operation='hitk'. Default : [2,5,10]
        A list of percentages at which to calculate the percent scores, ie the amount of indexes below this percentage.
        If percent_list == "all" a full dict with the length of classes will be created.
        Percentages are given as integers, ie 50 means 50 %.
    """
    if operation != "mp_value":
        # Melt the input profiles to long format
        print("Generating correlation matrix")
        similarity_melted_df = metric_melt(
            df=profiles,
            features=features,
            metadata_features=meta_features,
            similarity_metric=similarity_metric,
            eval_metric=operation,
        )
    print("Now starting grit calculation")
    # Perform the input operation
    if operation == "grit":
        metric_result = grit(
            similarity_melted_df=similarity_melted_df,
            control_perts=grit_control_perts,
            profile_col=replicate_groups["profile_col"],
            replicate_group_col=replicate_groups["replicate_group_col"],
            replicate_summary_method=grit_replicate_summary_method,
        )

    return metric_result


def grit(
    similarity_melted_df: pd.DataFrame,
    control_perts: List[str],
    profile_col: str,
    replicate_group_col: str,
    replicate_summary_method: str = "mean",
) -> pd.DataFrame:
    r"""Calculate grit

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        a long pandas dataframe output from cytominer_eval.transform.metric_melt
    control_perts : list
        a list of control perturbations to calculate a null distribution
    profile_col : str
        the metadata column storing profile ids. The column can have unique or replicate
        identifiers.
    replicate_group_col : str
        the metadata column indicating a higher order structure (group) than the
        profile column. E.g. target gene vs. guide in a CRISPR experiment.
    replicate_summary_method : {'mean', 'median'}, optional
        how replicate z-scores to control perts are summarized. Defaults to "mean".

    Returns
    -------
    pandas.DataFrame
        A dataframe of grit measurements per perturbation
    """
    # Check if we support the provided summary method
    # Determine pairwise replicates
    similarity_melted_df = assign_replicates(
        similarity_melted_df=similarity_melted_df,
        replicate_groups=[profile_col, replicate_group_col],
    )

    # Check to make sure that the melted dataframe is full
    assert_melt(similarity_melted_df, eval_metric="grit")

    # Extract out specific columns
    pair_ids = set_pair_ids()
    profile_col_name = "{x}{suf}".format(
        x=profile_col, suf=pair_ids[list(pair_ids)[0]]["suffix"]
    )

    # Define the columns to use in the calculation
    column_id_info = set_grit_column_info(
        profile_col=profile_col, replicate_group_col=replicate_group_col
    )

    # Calculate grit for each perturbation
    grit_df = (
        similarity_melted_df.groupby(profile_col_name)
        .apply(
            lambda x: calculate_grit(
                replicate_group_df=x,
                control_perts=control_perts,
                column_id_info=column_id_info,
                replicate_summary_method=replicate_summary_method,
            )
        )
        .reset_index(drop=True)
    )

    return grit_df


def calculate_grit(
    replicate_group_df: pd.DataFrame,
    control_perts: List[str],
    column_id_info: dict,
    distribution_compare_method: str = "zscore",
    replicate_summary_method: str = "mean",
) -> pd.Series:
    """Given an elongated pairwise correlation dataframe of replicate groups,
    calculate grit.

    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.grit.grit`.

    Parameters
    ----------
    replicate_group_df : pandas.DataFrame
        An elongated dataframe storing pairwise correlations of all profiles to a single
        replicate group.
    control_perts : list
        The profile_ids that should be considered controls (the reference)
    column_id_info: dict
        A dictionary of column identifiers noting profile and replicate group ids. This
        variable is autogenerated in
        :py:func:`cytominer_eval.transform.util.set_grit_column_info`.
    distribution_compare_method : {'zscore'}, optional
        How to compare the replicate and reference distributions of pairwise similarity
    replicate_summary_method : {'mean', 'median'}, optional
        How to summarize replicate z-scores. Defaults to "mean".

    Returns
    -------
    dict
        A return bundle of identifiers (perturbation, group) and results (grit score).
        The dictionary has keys ("perturbation", "group", "grit_score"). "grit_score"
        will be NaN if no other profiles exist in the defined group.
    """
    # Confirm that we support the user provided methods
    group_entry = get_grit_entry(replicate_group_df, column_id_info["group"]["id"])
    pert = get_grit_entry(replicate_group_df, column_id_info["profile"]["id"])

    # Define distributions for control perturbations
    control_distrib = replicate_group_df.loc[
        replicate_group_df.loc[:, column_id_info["profile"]["comparison"]].isin(
            control_perts
        ),
        "similarity_metric",
    ].values.reshape(-1, 1)

    assert len(control_distrib) > 1, "Error! No control perturbations found."

    # Define distributions for same group (but not same perturbation)
    same_group_distrib = replicate_group_df.loc[
        (
            replicate_group_df.loc[:, column_id_info["group"]["comparison"]]
            == group_entry
        )
        & (replicate_group_df.loc[:, column_id_info["profile"]["comparison"]] != pert),
        "similarity_metric",
    ].values.reshape(-1, 1)

    return_bundle = {"perturbation": pert, "group": group_entry}
    if len(same_group_distrib) == 0:
        return_bundle["grit"] = np.nan

    else:
        grit_score = compare_distributions(
            target_distrib=same_group_distrib,
            control_distrib=control_distrib,
            method=distribution_compare_method,
            replicate_summary_method=replicate_summary_method,
        )

        return_bundle["grit"] = grit_score

    return pd.Series(return_bundle)

def convert_pandas_dtypes(df: pd.DataFrame, col_fix: type = float) -> pd.DataFrame:
    r"""Helper funtion to convert pandas column dtypes

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to convert columns
    col_fix : {float, str}, optional
        A column type to convert the input dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with converted columns
    """
    try:
        df = df.astype(col_fix)
    except ValueError:
        raise ValueError(
            "Columns cannot be converted to {col}; check input features".format(
                col=col_fix
            )
        )

    return df


def get_grit_entry(df: pd.DataFrame, col: str) -> str:
    """Helper function to define the perturbation identifier of interest

    Grit must be calculated using unique perturbations. This may or may not mean unique
    perturbations.
    """
    entries = df.loc[:, col]
    #print(entries.unique())
    assert (
        len(entries.unique()) == 1
    ), "grit is calculated for each perturbation independently"
    return str(list(entries)[0])




def set_pair_ids():
    r"""Helper function to ensure consistent melted pairiwise column names

    Returns
    -------
    collections.OrderedDict
        A length two dictionary of suffixes and indeces of two pairs.
    """
    pair_a = "pair_a"
    pair_b = "pair_b"

    return_dict = OrderedDict()
    return_dict[pair_a] = {
        "index": "{pair_a}_index".format(pair_a=pair_a),
        "suffix": "_{pair_a}".format(pair_a=pair_a),
    }
    return_dict[pair_b] = {
        "index": "{pair_b}_index".format(pair_b=pair_b),
        "suffix": "_{pair_b}".format(pair_b=pair_b),
    }

    return return_dict

def assign_replicates(
    similarity_melted_df: pd.DataFrame,
    replicate_groups: List[str],
) -> pd.DataFrame:
    """Determine which profiles should be considered replicates.

    Given an elongated pairwise correlation matrix with metadata annotations, determine
    how to assign replicate information.

    Parameters
    ----------
    similarity_melted_df : pandas.DataFrame
        Long pandas DataFrame of annotated pairwise correlations output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : list
        a list of metadata column names in the original profile dataframe used to
        indicate replicate profiles.

    Returns
    -------
    pd.DataFrame
        A similarity_melted_df but with added columns indicating whether or not the
        pairwise similarity metric is comparing replicates or not. Used in most eval
        operations.
    """
    pair_ids = set_pair_ids()
    replicate_col_names = {x: "{x}_replicate".format(x=x) for x in replicate_groups}

    compare_dfs = []
    for replicate_col in replicate_groups:
        replicate_cols_with_suffix = [
            "{col}{suf}".format(col=replicate_col, suf=pair_ids[x]["suffix"])
            for x in pair_ids
        ]

        assert all(
            [x in similarity_melted_df.columns for x in replicate_cols_with_suffix]
        ), "replicate_group not found in melted dataframe columns"

        replicate_col_name = replicate_col_names[replicate_col]

        compare_df = similarity_melted_df.loc[:, replicate_cols_with_suffix]
        compare_df.loc[:, replicate_col_name] = False

        compare_df.loc[
            np.where(compare_df.iloc[:, 0] == compare_df.iloc[:, 1])[0],
            replicate_col_name,
        ] = True
        compare_dfs.append(compare_df)

    compare_df = pd.concat(compare_dfs, axis="columns").reset_index(drop=True)
    compare_df = compare_df.assign(
        group_replicate=compare_df.loc[:, replicate_col_names.values()].min(
            axis="columns"
        )
    ).loc[:, list(replicate_col_names.values()) + ["group_replicate"]]

    similarity_melted_df = similarity_melted_df.merge(
        compare_df, left_index=True, right_index=True
    )
    return similarity_melted_df


def assert_melt(
    df: pd.DataFrame, eval_metric: str = "replicate_reproducibility"
) -> None:
    r"""Helper function to ensure that we properly melted the pairwise correlation
    matrix

    Downstream functions depend on how we process the pairwise correlation matrix. The
    processing is different depending on the evaluation metric.

    Parameters
    ----------
    df : pandas.DataFrame
        A melted pairwise correlation matrix
    eval_metric : str
        The user input eval metric

    Returns
    -------
    None
        Assertion will fail if we incorrectly melted the matrix
    """

    pair_ids = set_pair_ids()
    df = df.loc[:, [pair_ids[x]["index"] for x in pair_ids]]
    index_sums = df.sum().tolist()

    assert_error = "Stop! The eval_metric provided in 'metric_melt()' is incorrect!"
    assert_error = "{err} This is a fatal error providing incorrect results".format(
        err=assert_error
    )
    if eval_metric == "replicate_reproducibility":
        assert index_sums[0] != index_sums[1], assert_error
    elif eval_metric == "precision_recall":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "grit":
        assert index_sums[0] == index_sums[1], assert_error
    elif eval_metric == "hitk":
        assert index_sums[0] == index_sums[1], assert_error



def set_grit_column_info(profile_col: str, replicate_group_col: str) -> dict:
    """Transform column names to be used in calculating grit

    In calculating grit, the data must have a metadata feature describing the core
    replicate perturbation (profile_col) and a separate metadata feature(s) describing
    the larger group (replicate_group_col) that the perturbation belongs to (e.g. gene,
    MOA).

    Parameters
    ----------
    profile_col : str
        the metadata column storing profile ids. The column can have unique or replicate
        identifiers.
    replicate_group_col : str
        the metadata column indicating a higher order structure (group) than the
        profile column. E.g. target gene vs. guide in a CRISPR experiment.

    Returns
    -------
    dict
        A nested dictionary of renamed columns indicating how to determine replicates
    """
    # Identify column transform names
    pair_ids = set_pair_ids()

    profile_id_with_suffix = [
        "{col}{suf}".format(col=profile_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    group_id_with_suffix = [
        "{col}{suf}".format(col=replicate_group_col, suf=pair_ids[x]["suffix"])
        for x in pair_ids
    ]

    col_info = ["id", "comparison"]
    profile_id_info = dict(zip(col_info, profile_id_with_suffix))
    group_id_info = dict(zip(col_info, group_id_with_suffix))

    column_id_info = {"profile": profile_id_info, "group": group_id_info}
    return column_id_info



def compare_distributions(
    target_distrib: List[float],
    control_distrib: List[float],
    method: str = "zscore",
    replicate_summary_method: str = "mean",
) -> float:
    """Compare two distributions and output a single score indicating the difference.

    Given two different vectors of distributions and a comparison method, determine how
    the two distributions are different.

    Parameters
    ----------
    target_distrib : np.array
        A list-like (e.g. numpy.array) of floats representing the first distribution.
        Must be of shape (n_samples, 1).
    control_distrib : np.array
        A list-like (e.g. numpy.array) of floats representing the second distribution.
        Must be of shape (n_samples, 1).
    method : str, optional
        A string indicating how to compare the two distributions. Defaults to "zscore".
    replicate_summary_method : str, optional
        A string indicating how to summarize the resulting scores, if applicable. Only
        in use when method="zscore".

    Returns
    -------
    float
        A single value comparing the two distributions
    """
    # Confirm that we support the provided methods

    if method == "zscore":
        scaler = StandardScaler()
        scaler.fit(control_distrib)
        scores = scaler.transform(target_distrib)

        if replicate_summary_method == "mean":
            scores = np.mean(scores)
        elif replicate_summary_method == "median":
            scores = np.median(scores)

    return scores

def assert_pandas_dtypes(df: pd.DataFrame, col_fix: type = float) -> pd.DataFrame:
    r"""Helper funtion to ensure pandas columns have compatible columns

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to convert columns
    col_fix : {float, str}, optional
        A column type to convert the input dataframe.

    Returns
    -------
    pd.DataFrame
        A dataframe with converted columns
    """
    assert col_fix in [str, float], "Only str and float are supported"

    df = convert_pandas_dtypes(df=df, col_fix=col_fix)

    assert_error = "Columns not successfully updated, is the dataframe consistent?"
    if col_fix == str:
        assert all([ptypes.is_string_dtype(df[x]) for x in df.columns]), assert_error

    if col_fix == float:
        assert all([ptypes.is_numeric_dtype(df[x]) for x in df.columns]), assert_error

    return df

def process_melt(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    eval_metric: str = "replicate_reproducibility",
) -> pd.DataFrame:
    """Helper function to annotate and process an input similarity matrix

    Parameters
    ----------
    df : pandas.DataFrame
        A similarity matrix output from
        :py:func:`cytominer_eval.transform.transform.get_pairwise_metric`
    meta_df : pandas.DataFrame
        A wide matrix of metadata information where the index aligns to the similarity
        matrix index
    eval_metric : str, optional
        Which metric to ultimately calculate. Determines whether or not to keep the full
        similarity matrix or only one diagonal. Defaults to "replicate_reproducibility".

    Returns
    -------
    pandas.DataFrame
        A pairwise similarity matrix
    """
    # Confirm that the user formed the input arguments properly
    assert df.shape[0] == df.shape[1], "Matrix must be symmetrical"

    # Get identifiers for pairing metadata
    pair_ids = set_pair_ids()
    
    # Subset the pairwise similarity metric depending on the eval metric given:
    #   "replicate_reproducibility" - requires only the upper triangle of a symmetric matrix
    #   "precision_recall" - requires the full symmetric matrix (no diagonal)
    # Remove pairwise matrix diagonal and redundant pairwise comparisons
    if eval_metric == "replicate_reproducibility":
        upper_tri = get_upper_matrix(df)
        df = df.where(upper_tri)
    else:
        np.fill_diagonal(df.values, np.nan)
    print("Melting to long")
    # Convert pairwise matrix to melted (long) version based on index value
    metric_unlabeled_df = (
        pd.melt(
            df.reset_index(),
            id_vars="index",
            value_vars=df.columns,
            var_name=pair_ids["pair_b"]["index"],
            value_name="similarity_metric",
        )
        .dropna()
        .reset_index(drop=True)
        .rename({"index": pair_ids["pair_a"]["index"]}, axis="columns")
    )
    print("Merging meta with correlations")
    # Merge metadata on index for both comparison pairs
    output_df = meta_df.merge(
        meta_df.merge(
            metric_unlabeled_df,
            left_index=True,
            right_on=pair_ids["pair_b"]["index"],
        ),
        left_index=True,
        right_on=pair_ids["pair_a"]["index"],
        suffixes=[pair_ids["pair_a"]["suffix"], pair_ids["pair_b"]["suffix"]],
    ).reset_index(drop=True)

    return output_df


def metric_melt(
    df: pd.DataFrame,
    features: List[str],
    metadata_features: List[str],
    eval_metric: str = "replicate_reproducibility",
    similarity_metric: str = "pearson",
) -> pd.DataFrame:
    """Helper function to fully transform an input dataframe of metadata and feature
    columns into a long, melted dataframe of pairwise metric comparisons between
    profiles.

    Parameters
    ----------
    df : pandas.DataFrame
        A profiling dataset with a mixture of metadata and feature columns
    features : list
        Which features make up the profile; included in the pairwise calculations
    metadata_features : list
        Which features are considered metadata features; annotate melted dataframe and
        do not use in pairwise calculations.
    eval_metric : str, optional
        Which metric to ultimately calculate. Determines whether or not to keep the full
        similarity matrix or only one diagonal. Defaults to "replicate_reproducibility".
    similarity_metric : str, optional
        The pairwise comparison to calculate

    Returns
    -------
    pandas.DataFrame
        A fully melted dataframe of pairwise correlations and associated metadata
    """
    # Subset dataframes to specific features
    df = df.reset_index(drop=True)

    assert all(
        [x in df.columns for x in metadata_features]
    ), "Metadata feature not found"
    assert all([x in df.columns for x in features]), "Profile feature not found"

    meta_df = df.loc[:, metadata_features]
    df = df.loc[:, features]

    # Convert pandas column types and assert conversion success
    meta_df = assert_pandas_dtypes(df=meta_df, col_fix=str)
    df = assert_pandas_dtypes(df=df, col_fix=float)
    
    print("Now calculating corelation matrix")
    # Get pairwise metric matrix
    pair_df = get_pairwise_metric(df=df, similarity_metric=similarity_metric)
    
    print("Now melting datafdrame")
    # Convert pairwise matrix into metadata-labeled melted matrix
    output_df = process_melt(df=pair_df, meta_df=meta_df, eval_metric=eval_metric)

    return output_df


def get_pairwise_metric(df: pd.DataFrame, similarity_metric: str) -> pd.DataFrame:
    """Helper function to output the pairwise similarity metric for a feature-only
    dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples x features, where all columns can be coerced to floats
    similarity_metric : str
        The pairwise comparison to calculate

    Returns
    -------
    pandas.DataFrame
        A pairwise similarity matrix
    """
    df = assert_pandas_dtypes(df=df, col_fix=float)
    print(df.shape)

    pair_df = df.transpose().corr(method=similarity_metric)

    # Check if the metric calculation went wrong
    # (Current pandas version makes this check redundant)
    if pair_df.shape == (0, 0):
        raise TypeError(
            "Something went wrong - check that 'features' are profile measurements"
        )

    return pair_df



def get_upper_matrix(df: pl.DataFrame, batch_size = 10000) -> np.ndarray:
    """
    Generate an upper triangle mask for a large DataFrame in batches.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame for which the upper triangle mask is to be created.
    batch_size : int
        The size of each batch.

    Returns
    -------
    np.ndarray
        An upper triangle matrix the same shape as the input DataFrame.
    """
    nrows, ncols = df.shape
    upper_matrix = np.zeros((nrows, ncols), dtype=bool)
    
    for start_row in range(0, nrows, batch_size):
        end_row = min(start_row + batch_size, nrows)
        for start_col in range(0, ncols, batch_size):
            end_col = min(start_col + batch_size, ncols)
            
            # Create a mask for the current batch
            batch_mask = np.triu(np.ones((end_row - start_row, end_col - start_col)), k=1).astype(bool)
            
            # Place the batch mask in the corresponding position of the full matrix
            upper_matrix[start_row:end_row, start_col:end_col] = batch_mask

    return upper_matrix