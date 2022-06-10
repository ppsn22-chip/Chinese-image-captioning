
def get_image_path(new_path):
    global predicted_path
    print("wait a new path...")
    # =========== 2.0 版本==============
    while(True):

        time.sleep(5)
        paths = {}
        with open(new_path,'r') as f:

            try:
                paths = json.load(f)
            except json.decoder.JSONDecodeError:
                print("no new path...")
                continue

            paths_list = list(paths.keys())
            # print(paths_list)

            for _path in paths_list:
                if paths[_path] is False and (_path not in predicted_path):
                    insert_path(_path)
                    return _path

# 创建固定长度的已预测缓存
predicted_path = []

def insert_path(predict_path):
    global predicted_path
    # 缓存长度200
    if len(predicted_path) == 200:
        predicted_path.pop(0)
        predicted_path.append(predict_path)
    else:
        predicted_path.append(predict_path)