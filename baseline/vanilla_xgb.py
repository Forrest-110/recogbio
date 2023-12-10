import xgboost as xgb
import numpy as np
def load_dataset():
    root = "dataset/ivis/"
    train_dir = root + "train/"
    test_dir = root + "test/"

    y_train = np.load(train_dir + "y_task_0.npy")
    y_test = np.load(test_dir + "y_task_0.npy")

    rsfc_atlas_feat_train = np.load(train_dir + "0_task_0.npy")
    rsfc_atlas_feat_test = np.load(test_dir + "0_task_0.npy")

    rsfc_yeo_feat_train = np.load(train_dir + "1_task_0.npy")
    rsfc_yeo_feat_test = np.load(test_dir + "1_task_0.npy")

    scfp_atlas_feat_train = np.load(train_dir + "2_task_0.npy")
    scfp_atlas_feat_test = np.load(test_dir + "2_task_0.npy")

    print("Dataset Info:")
    print("rsfc_atlas_feat_train.shape: ", rsfc_atlas_feat_train.shape)
    print("rsfc_atlas_feat_test.shape: ", rsfc_atlas_feat_test.shape)
    print("rsfc_yeo_feat_train.shape: ", rsfc_yeo_feat_train.shape)
    print("rsfc_yeo_feat_test.shape: ", rsfc_yeo_feat_test.shape)
    print("scfp_atlas_feat_train.shape: ", scfp_atlas_feat_train.shape)
    print("scfp_atlas_feat_test.shape: ", scfp_atlas_feat_test.shape)
    print("y_train.shape: ", y_train.shape)
    print("y_test.shape: ", y_test.shape)

    data={
        "rsfc_atlas_feat_train": rsfc_atlas_feat_train,
        "rsfc_atlas_feat_test": rsfc_atlas_feat_test,
        "rsfc_yeo_feat_train": rsfc_yeo_feat_train,
        "rsfc_yeo_feat_test": rsfc_yeo_feat_test,
        "scfp_atlas_feat_train": scfp_atlas_feat_train,
        "scfp_atlas_feat_test": scfp_atlas_feat_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return data



def xgbconfig():
    params=dict()
    params["tree_method"] = "hist"
    return params

def train(params, data):
    rsfc_atlas_feat_train = data["rsfc_atlas_feat_train"]
    rsfc_atlas_feat_test = data["rsfc_atlas_feat_test"]
    rsfc_yeo_feat_train = data["rsfc_yeo_feat_train"]
    rsfc_yeo_feat_test = data["rsfc_yeo_feat_test"]
    scfp_atlas_feat_train = data["scfp_atlas_feat_train"]
    scfp_atlas_feat_test = data["scfp_atlas_feat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train = np.concatenate([rsfc_atlas_feat_train, rsfc_yeo_feat_train, scfp_atlas_feat_train], axis=1)
    X_test = np.concatenate([rsfc_atlas_feat_test, rsfc_yeo_feat_test, scfp_atlas_feat_test], axis=1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, "test")])

    return bst

def test(bst, data):
    rsfc_atlas_feat_test = data["rsfc_atlas_feat_test"]
    rsfc_yeo_feat_test = data["rsfc_yeo_feat_test"]
    scfp_atlas_feat_test = data["scfp_atlas_feat_test"]
    y_test = data["y_test"]

    X_test = np.concatenate([rsfc_atlas_feat_test, rsfc_yeo_feat_test, scfp_atlas_feat_test], axis=1)
    dtest = xgb.DMatrix(X_test, label=y_test)

    preds = bst.predict(dtest)
    return preds

def pearson_corr(preds, y_test):
    # preds: N x 1
    # y_test: N x 1
    preds = preds - np.mean(preds)
    y_test = y_test - np.mean(y_test)
    corr = np.sum(preds * y_test) / (np.sqrt(np.sum(preds ** 2)) * np.sqrt(np.sum(y_test ** 2)))
    return corr

def evaluate(preds, data):
    y_test = data["y_test"]
    
    # preds: N x 1
    # y_test: N x 1
    
    # mae loss
    mae_loss= np.mean(np.abs(preds - y_test))
    # corr
    corr = pearson_corr(preds, y_test)

    print("MAE Loss: ", mae_loss)

    print("Corr: ", corr)

    return mae_loss, corr

def main():
    data = load_dataset()
    params = xgbconfig()
    bst = train(params, data)
    preds = test(bst, data)
    evaluate(preds, data)

if __name__ == "__main__":
    main()
