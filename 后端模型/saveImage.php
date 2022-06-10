<?php

define("STORE_PATH","/var/www/html/");


$ret['status'] = "false";
$ret['info'] = "null";
$ret['file_type'] = "null";
$ret['Caption'] = "null";
$ret['file_name'] = "null";


if (isset($_FILES['myFile'])) {


    // echo "tmp name:", $_FILES['myFile']['tmp_name'],"\n";
    // echo "file name ",$_FILES['myFile']['name'],"\n";
    // echo "file type:",$_FILES["myFile"]["type"],"\n";
    // echo "file size:",$_FILES["myFile"]["size"],"\n";

    // allowed file type
    $allowedExts = array("jpg");

    // get uploaded file type
    $temp = explode(".", $_FILES["myFile"]["name"]);
    $extension = end($temp);        // 获取文件后缀名

    $if_succ = false;


    // only jpg file
    if (in_array($extension,$allowedExts)){

        if ($_FILES["myFile"]["error"] > 0)
        {
            $ret['status'] = "failed";
            $ret['info'] = $_FILES["myFile"]["error"];
            //echo "错误：: " . $_FILES["myFile"]["error"] . "<br>";
        }
        else
        {
            # echo "上传文件名: " . $_FILES["myFile"]["name"] . "<br>";
            # echo "文件类型: " . $_FILES["myFile"]["type"] . "<br>";
            # echo "文件大小: " . ($_FILES["myFile"]["size"] / 1024) . " kB<br>";
            # echo "文件临时存储的位置: " . $_FILES["myFile"]["tmp_name"];
            
            if (file_exists(STORE_PATH . $_FILES["myFile"]["name"])){
                echo $_FILES["myFile"]["name"] . " 文件已经存在。 ";
            }
            // 如果 upload 目录不存在该文件则将文件上传到 upload 目录下;
            else{
                
                $if_succ = move_uploaded_file($_FILES["myFile"]["tmp_name"], STORE_PATH . $_FILES["myFile"]["name"]);
                //echo "我们的文件存储在: " . STORE_PATH . $_FILES["myFile"]["name"];
                
                //$ret['file_store'] = STORE_PATH . $_FILES["myFile"]["name"];

                $image_path = "/var/www/html/" . $_FILES["myFile"]["name"];
                //echo "我们准备把照片传给py";

                // 执行外部程序获取结果
                if ($if_succ){
                    //  echo "我们的目标是胜利";
                    //  echo $image_path;
                     //exec("python3 app.py);
                     exec("python3 getCaption.py {$image_path} 2>error.log",$out,$res);
                     //echo "在getCaption这一步了";
                }
                else{
                    $ret['info'] = '文件上传失败';
                }
                // echo $out[0];
                // echo $out[1];
                // echo $out[2];
                $ret = ["Caption"=>$out[2],"file_name"=>$_FILES['myFile']['name']];
                // 
                
            }
        }
    }
    else{

        $ret['status'] = "failed";
        $ret['info'] = "非法的文件格式";
        $ret['file_type'] = $_FILES["myFile"]["type"];
    }


    // $predict =  get_predict(predictions_json_path)
    if ($res==0){

        $ret['status'] = "succ";
        $ret['info'] = "成功获取预测结果";
        
    }
    else{
        $ret['info'] = "获取预测结果失败";
    }

    
}
else{
    $ret['info'] = "文件为空";
}
echo json_encode($ret,JSON_UNESCAPED_UNICODE);

?>
