# this program is used for php call
import os
import json
import datetime
import time
import random
import sys


new_path = "/var/www/html/SeeingServer/predict_result"
share_path = "/var/www/html/share_path.json"
result_path = os.path.join(new_path,"predict.json")

# 此文件从多模型返回值继承，所以需要设置模型个数
num_models = 1
model_name = "CN_SoftAtt"

def getCaption(image_path):
    #print("url path 来自参数",url_path)
    print("存储地址，获取结果")
    # image_path = "/var/www/html"
    # global img_text
    paths_bool = {}
    
    with open(share_path,'r+') as share_file:
        # print("open share path file..")
        # if notEmpty:
        
        try:
            paths_bool =  json.load(share_file)
            # print("file with..",paths_bool)
            paths_bool[image_path] = False

        except json.decoder.JSONDecodeError:
            print("地址文件解析失败，可能为空，直接写入")
            paths_bool = {image_path:False}
        
        share_file.seek(0)
        paths_bool.update(paths_bool)
        # print("save new path:",session.get("Gurl_path"))
        json.dump(paths_bool,share_file)
        print("已将图像地址存储到共享文件中.")

    t1 = datetime.datetime.now()
    while(True):

        img_text = {}
        
        notEmpty = False
        time.sleep(random.random())#每隔一秒执行一次
        with open(result_path,'r') as result_file:
            if bool(result_file.read()):
                notEmpty = True
        if notEmpty:
            with open(result_path,'r') as result_file:
                for line in result_file.readlines():
                    
                    img_text = json.loads(line)
                    # print(img_text)
                #确认已经返回了对应图片的答案并且是两个模型再进行进一步操作
                if image_path in img_text.keys() and len(img_text[image_path].keys()) == num_models:          
                    Caption = img_text[image_path][model_name]
                    print(Caption)
                    #return Caption
                    # print("SCST:",session['img_text'][session.get("Gurl_path")][models_name[2]])
                    # TODO 也删除预测出来的结果,以增快响应时间
                    deletePath(share_path, image_path)
                    break
        t2 = datetime.datetime.now()
        if (t2-t1).seconds > 20:
            print("无法正确从文件中获取，放弃，仍删除路径")
            deletePath(share_path, image_path)
            break
        # print("尝试从文件中读取结果，对应的键值：",session.get("Gurl_path"))
        # print("session 中的内容：",session.get("file_name"))
    
def deletePath(share_path, del_path):
    with open(share_path,'r+') as share_file:
        # print("delete path...")
        # if notEmpty:
        
        try:
            paths_bool =  json.load(share_file)
            # del paths_bool[del_path]
            paths_bool[del_path] = True
            share_file.seek(0)
            share_file.truncate()
            paths_bool.update(paths_bool)
            json.dump(paths_bool,share_file)
        except json.decoder.JSONDecodeError:
            print("Error: file is null when del the key path:",del_path)

        print("将地址键值置为True",del_path)

if __name__ == "__main__":

 
    image_path = sys.argv[1]
    getCaption(image_path)