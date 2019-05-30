import sklearn.model_selection
import sklearn.metrics
import numpy as np
import autosklearn.regression
import pandas as pd

def rmse(solution, prediction):
    # custom function defining accuracy
    #return np.sqrt(((prediction - solution) ** 2).mean())
    return np.sqrt(sklearn.metrics.mean_squared_error(solution, prediction))


def main():
    f = open("/home/lazafi/labor/ml-2019/male2019/2/data/bikesharing/bikesharing_clean.csv")
    f.readline()  # skip the header
    #data = np.genfromtxt(f, delimiter=",", dtype="S5,i8,i8,i8,S5,i8,i8,S5,f8,f8,f8,f8,i8")
    data = np.genfromtxt(f, delimiter=",")
    X = data[:, 1:12]
    print(X)
    y = data[:, 12] 
    print(y)
    feature_types =  (['categorical'] * 7) + (['numerical'] * 4)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
#        time_left_for_this_task=7200,
#        time_left_for_this_task=90,
#        per_run_time_limit=10,
        tmp_folder='/tmp/autosklearn_bikesharing2_tmp',
        output_folder='/tmp/autosklearn_bikesharing2_out',
        n_jobs=4,
        seed=5,
 #       exclude_estimators='DUMMY',
        delete_output_folder_after_terminate=False,
        delete_tmp_folder_after_terminate=False,
        ml_memory_limit=6144,
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}, 
    )



    print(X)


    rmse_scorer = autosklearn.metrics.make_scorer(
        name="rmse",
        score_func=rmse,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )

    mse_scorer = autosklearn.metrics.make_scorer(
        name="mse",
        score_func= sklearn.metrics.mean_squared_error,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )


    print("start fitting")
    automl.fit(X_train, y_train, dataset_name='bikesharing', feat_type=feature_types, metric=rmse_scorer)
    print("end fittin")
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("===")
    print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    print("MSE score:", sklearn.metrics.mean_squared_error(y_test, predictions))
    print("RMSE score:", rmse(y_test, predictions))
    print("###")
    print(automl.sprint_statistics())

if __name__ == '__main__':
    main()
