import sys
sys.path.append('/home/yanzhenghang/pythonRemote/kaggle_script')
import imet.make_folds as immfd
import imet.main as imm
import imet.make_submission as immsb


# immfd.main()
# imm.main('train', 'model_3')
# imm.main('predict_test', 'model_3')
immsb.main(['model_3/test.h5','model_1/test.h5'], 'submission.csv')



pass


