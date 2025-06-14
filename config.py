import os


'''文件路径'''
# 数据集路径
TRAIN_JSON_PATH = r"gossipcop_train_set.json"  # 替换为实际路径
DEV_JSON_PATH = r"analyze_val_ch.json"      # 替换为实际路径
TEST_JSON_PATH = r"weibo21_test_newsummary.json"    # 替换为实际路径

# 预训练BERT模型路径
BERT_PATH = "chinese_roberta_wwm_ext"       # 替换为实际chinese_roberta_wwm_ext路径

# 模型保存文件夹
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model_saved')
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


'''模型配置'''
EPOCHS = 6
BATCH_SIZE = 32