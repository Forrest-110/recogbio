from ivis import Ivis
import numpy as np
data_path="/Datasets/recogbio/"
output_path="dataset/ivis/"

gt_train=np.loadtxt(data_path+'train/gt.csv', delimiter=',')
gt_test=np.loadtxt(data_path+'test/gt.csv', delimiter=',')

rsfc_atlas_feat_train=np.loadtxt(data_path+'train/rsfc_atlas_feat.csv', delimiter=',')
rsfc_atlas_feat_test=np.loadtxt(data_path+'test/rsfc_atlas_feat.csv', delimiter=',')
rsfc_yeo_feat_train=np.loadtxt(data_path+'train/rsfc_yeo_feat.csv', delimiter=',')
rsfc_yeo_feat_test=np.loadtxt(data_path+'test/rsfc_yeo_feat.csv', delimiter=',')
scfp_atlas_feat_train=np.loadtxt(data_path+'train/scfp_atlas_feat.csv', delimiter=',')
scfp_atlas_feat_test=np.loadtxt(data_path+'test/scfp_atlas_feat.csv', delimiter=',')

pred_task_num=gt_train.shape[1]
orig_dims=rsfc_atlas_feat_train.shape[1]
latent_dims=128

X_trains = [rsfc_atlas_feat_train, rsfc_yeo_feat_train, scfp_atlas_feat_train]
X_tests = [rsfc_atlas_feat_test, rsfc_yeo_feat_test, scfp_atlas_feat_test]

supervision_metric = 'mae'
ivis = Ivis(embedding_dims=latent_dims,k=15, batch_size=32, supervision_metric=supervision_metric,supervision_weight=0.9)

for i in range(52,pred_task_num):
    print('Training for task %d' % i)

    for j in range(len(X_trains)):
        ivis.fit(X_trains[j], gt_train[:,i])
        print('Saving for task %d' % i)
        ivis.save_model(output_path+'dim_reduce_models/%d_task_%d.h5' % (j,i))
        X_train_new = ivis.transform(X_trains[j])
        print('Transforming test data for task %d' % i)
        X_test_new = ivis.transform(X_tests[j])
        print('Saving transformed train data for task %d' % i)
        np.save(output_path+'train/%d_task_%d.npy' % (j,i), X_train_new)
        print('Saving transformed test data for task %d' % i)
        np.save(output_path+'test/%d_task_%d.npy' % (j,i), X_test_new)
    print('Saving y_train for task %d' % i)
    np.save(output_path+'train/y_task_%d.npy' % i, gt_train[:,i])
    print('Saving y_test for task %d' % i)
    np.save(output_path+'test/y_task_%d.npy' % i, gt_test[:,i])

    

