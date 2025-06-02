import click
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
import os

@click.command()
@click.option('input_dataset', '-i', type=click.Path(exists=True))
@click.option('input_valid', '-v', type=click.Path(exists=True))
@click.option('input_test', '-t', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, help='Folder for results.')
def train_model(input_dataset, input_valid, input_test, output):
    # Determine label codes based on input file name
    if "BF" in output:
        label_codes = {
            8: 'retinoid receptor agonist',
            9: 'topoisomerase inhibitor',
            0: 'ATPase inhibitor',
            10: 'tubulin polymerization inhibitor',
            6: 'dmso',
            7: 'protein synthesis inhibitor',
            5: 'PARP inhibitor',
            1: 'Aurora kinase inhibitor',
            3: 'HSP inhibitor',
            2: 'HDAC inhibitor',
            4: 'JAK inhibitor'
        }
    elif "celldeath" in input_dataset.lower():
        print("CP labels")
        label_codes = {
                        2: "ferroptosis inducer",
                        4: "necroptosis inducer",
                        0: "apoptosis",
                        1: "autophagy inducer",
                        3: "immunogenic cell death",
                        5: "pyroptosis inducer"
                    }
    else:
        label_codes = {
            0: "AKT",
            1: "CDK",
            2: "DMSO",
            3: "HDAC",
            4: "MAPK",
            5: "PARP",
            6: "TUB"
        }

    # Load the datasets
    df_train = pd.read_csv(input_dataset)
    df_val = pd.read_csv(input_valid)
    df_test = pd.read_csv(input_test)

    # Normalize the data
    scaler = StandardScaler()
    X_train = df_train.drop(['label'], axis=1).values
    X_val = df_val.drop(['label'], axis=1).values
    X_test = df_test.drop(['label'], axis=1).values

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create DataFrames for AutoGluon    
    #train_data = pd.DataFrame(X_train_scaled, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    train_data = df_train
    #train_data['label'] = df_train['label'].values

    #tune_data = pd.DataFrame(X_val_scaled, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
    tune_data = df_val
    #tune_data['label'] = df_val['label'].values
    
    combined_data = pd.concat([train_data, tune_data])

    #test_data = pd.DataFrame(X_test_scaled, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_data = df_test
    y_test = df_test['label'].values

    # Train AutoGluon model using the validation set as a tuning set
    print("Training AutoGluon model...")
    time_limit = 86400  # 24 hours
    excluded_model_types = ['KNN']
    predictor = TabularPredictor(label='label', path = output).fit(
        train_data=combined_data,
        #tuning_data=tune_data,
        #use_bag_holdout=True,
        #ag_args_fit={'num_gpus': 1},
        presets='high_quality',
	    excluded_model_types=excluded_model_types,
        time_limit=time_limit,
        num_cpus=8 
        #num_folds_parallel=2
    )

    # Evaluate on the test set
    y_pred = predictor.predict(test_data)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=[label_codes[i] for i in np.unique(y_test)]))

    # Save the model
    predictor.save(f'{output}/autogluon_model')

if __name__ == '__main__':
    train_model()
