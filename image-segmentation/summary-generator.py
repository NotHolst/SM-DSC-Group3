from keras.models import load_model
import os.path
import os
import sys

folders = os.listdir('./Models/')
print(folders)


for folder in folders:
    if os.path.exists('./Models/' + folder + '/model.h5'):
        model = load_model('./Models/' + folder + '/model.h5')
        stdout_summary = sys.stdout
        with open('./Models/' + folder + '/summary.txt', 'w') as f:
            sys.stdout = f
            model.summary()
        sys.stdout = stdout_summary
    

