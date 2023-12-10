from sklearn import svm
import numpy as np
def load_dataset(need_y_train=True, need_y_test=True, need_rsfc_atlas_feat_train=True, need_rsfc_atlas_feat_test=True, need_rsfc_yeo_feat_train=True, need_rsfc_yeo_feat_test=True, need_scfp_atlas_feat_train=True, need_scfp_atlas_feat_test=True):
    root = "/Datasets/recogbio/"
    train_dir = root + "train/"
    test_dir = root + "test/"

    y_train = None
    y_test = None
    rsfc_atlas_feat_train = None
    rsfc_atlas_feat_test = None
    rsfc_yeo_feat_train = None
    rsfc_yeo_feat_test = None
    scfp_atlas_feat_train = None
    scfp_atlas_feat_test = None


    if need_y_train:
        y_train = np.loadtxt(train_dir + "gt.csv", delimiter=",")
    if need_y_test:
        y_test = np.loadtxt(test_dir + "gt.csv", delimiter=",")
    if need_rsfc_atlas_feat_train:
        rsfc_atlas_feat_train = np.loadtxt(train_dir + "rsfc_atlas_feat.csv", delimiter=",")
    if need_rsfc_atlas_feat_test:
        rsfc_atlas_feat_test = np.loadtxt(test_dir + "rsfc_atlas_feat.csv", delimiter=",")
    if need_rsfc_yeo_feat_train:
        rsfc_yeo_feat_train = np.loadtxt(train_dir + "rsfc_yeo_feat.csv", delimiter=",")
    if need_rsfc_yeo_feat_test:
        rsfc_yeo_feat_test = np.loadtxt(test_dir + "rsfc_yeo_feat.csv", delimiter=",")
    if need_scfp_atlas_feat_train:    
        scfp_atlas_feat_train = np.loadtxt(train_dir + "scfp_atlas_feat.csv", delimiter=",")
    if need_scfp_atlas_feat_test:
        scfp_atlas_feat_test = np.loadtxt(test_dir + "scfp_atlas_feat.csv", delimiter=",")

    # print("Dataset Info:")
    # print("rsfc_atlas_feat_train.shape: ", rsfc_atlas_feat_train.shape)
    # print("rsfc_atlas_feat_test.shape: ", rsfc_atlas_feat_test.shape)
    # print("rsfc_yeo_feat_train.shape: ", rsfc_yeo_feat_train.shape)
    # print("rsfc_yeo_feat_test.shape: ", rsfc_yeo_feat_test.shape)
    # print("scfp_atlas_feat_train.shape: ", scfp_atlas_feat_train.shape)
    # print("scfp_atlas_feat_test.shape: ", scfp_atlas_feat_test.shape)
    # print("y_train.shape: ", y_train.shape)
    # print("y_test.shape: ", y_test.shape)

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



def train(data, params=None):
    rsfc_atlas_feat_train = data["rsfc_atlas_feat_train"]
    rsfc_yeo_feat_train = data["rsfc_yeo_feat_train"]
    scfp_atlas_feat_train = data["scfp_atlas_feat_train"]
    y_train = data["y_train"][:,params["target"] if params else 0]
    X_train = np.concatenate((rsfc_atlas_feat_train, rsfc_yeo_feat_train, scfp_atlas_feat_train), axis=1)
    model=svm.SVR()
    model.fit(X_train, y_train)
    return model

def test(model, data, params=None):
    rsfc_atlas_feat_test = data["rsfc_atlas_feat_test"]
    rsfc_yeo_feat_test = data["rsfc_yeo_feat_test"]
    scfp_atlas_feat_test = data["scfp_atlas_feat_test"]
    y_test = data["y_test"][:,params["target"] if params else 0]

    X_test = np.concatenate((rsfc_atlas_feat_test, rsfc_yeo_feat_test, scfp_atlas_feat_test), axis=1)
    preds = model.predict(X_test)
    return preds, y_test

def evaluate(preds, y_test):
  
    # preds: N x 1
    # y_test: N x 1
    
    # L2 loss
    l2_loss = np.mean(np.square(preds - y_test), axis=0)
    print("L2 loss: ", l2_loss)
    # corr
    corr = np.corrcoef(preds, y_test)[0,1]

    print("Corr: ", corr)

    return l2_loss, corr

def main():
    data = load_dataset(need_y_test=True, need_y_train=False, need_rsfc_atlas_feat_train=False, need_rsfc_atlas_feat_test=True, need_rsfc_yeo_feat_train=False, need_rsfc_yeo_feat_test=True, need_scfp_atlas_feat_train=False, need_scfp_atlas_feat_test=True)
    # model = train(data)

    # #save model
    # import pickle
    # with open('svm_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    #load model
    import pickle
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    preds,y_pred = test(model, data)
    evaluate(preds, y_pred)

if __name__ == "__main__":
    main()
