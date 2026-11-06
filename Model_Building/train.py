from Model_Building.rfe_cv import run_rfe
from Model_Building.ModelBuilder import ModelBuilder
from params.generic_params import DO_NOT_MODEL
from params.targets import NIL_TARGET, TARGET_COL
import polars as pl 

def main(run_name="test_run"):

    # Setup model run 
    train = pl.read_parquet("/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train.parquet")

    features = [col for col in train.columns if col not in DO_NOT_MODEL]

    # run rfe for each model
    feature_selection_dict = {}
    for model in ["nil", "sev"]:
        target = NIL_TARGET if model == "nil" else TARGET_COL
        features = run_rfe(train, features, target, model)
        feature_selection_dict[model] = features
    
    # Run modelbuilder
    mb = ModelBuilder(
        nil_features=feature_selection_dict["nil"],
        nil_params={},
        sev_features=feature_selection_dict["sev"],
        sev_params={} 
    )

    mb.fit(train)

    mb.save(rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}.joblib")

if __name__ == "__main__":
    main()