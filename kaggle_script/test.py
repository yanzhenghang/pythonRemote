import sys
sys.path.append('/home/yanzhenghang/pythonRemote/kaggle_script')
import imet.make_folds as immfd
import imet.main as imm
import imet.make_submission as immsb


immfd.main()
# imm.main('train', 'model_1')
imm.main('predict_test', 'model_1')
immsb.main(['model_1/test.h5'], 'submission.csv')



pass


