import numpy as np

# trainset
samsung_train_ids = np.zeros(61162)
for i in range(61162):
    samsung_train_ids[i] = i+13407

samsung_train_ids = samsung_train_ids.astype(int)
np.save('coco_train_ids.npy', samsung_train_ids)

#valset
samsung_dev_ids = np.zeros(5000)
for i in range(5000):
    samsung_dev_ids[i] = i+1

samsung_train_ids = samsung_dev_ids.astype(int)
np.save('coco_dev_ids.npy', samsung_dev_ids)

#testset
samsung_test_ids = np.zeros(5000)
for i in range(5000):
    samsung_test_ids[i] = i+5001

samsung_train_ids = samsung_test_ids.astype(int)
np.save('coco_test_ids.npy', samsung_test_ids)

#valset 남은것
samsung_restval_ids= np.zeros(3406)
for i in range(3406):
    samsung_restval_ids[i] = i+10001

samsung_train_ids = samsung_restval_ids.astype(int)
np.save('coco_restval_ids.npy', samsung_restval_ids)