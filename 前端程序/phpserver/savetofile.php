// 本部分代码已经不作为整个项目的使用，APP后端+模型代码已经上传在GitHub的私有仓库，named  SeeingServer
<?php

define("STORE_PATH","uploads/");


if (isset($_FILES['myFile'])) {

    // $ret = array();
    $ret['file_name'] = $_FILES['myFile']['name'];


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
            echo "错误：: " . $_FILES["myFile"]["error"] . "<br>";
        }
        else
        {
            // echo "上传文件名: " . $_FILES["myFile"]["name"] . "<br>";
            // echo "文件类型: " . $_FILES["myFile"]["type"] . "<br>";
            // echo "文件大小: " . ($_FILES["myFile"]["size"] / 1024) . " kB<br>";
            // echo "文件临时存储的位置: " . $_FILES["myFile"]["tmp_name"];

            if (file_exists(STORE_PATH . $_FILES["myFile"]["name"])){
                echo $_FILES["myFile"]["name"] . " 文件已经存在。 ";
            }
            // 如果 upload 目录不存在该文件则将文件上传到 upload 目录下;
            else{

                $if_succ = move_uploaded_file($_FILES["myFile"]["tmp_name"], STORE_PATH . $_FILES["myFile"]["name"]);
                // echo "文件存储在: " . STORE_PATH . $_FILES["myFile"]["name"];

                $ret['file_store'] = STORE_PATH . $_FILES["myFile"]["name"];
            }
        }
    }
    else{
        echo "非法的文件格式" . "<br>";
        echo "文件类型: " . $_FILES["myFile"]["type"] . "<br>";
    }


    // to get the predictions of the pictuire

    // $predict =  get_predict(predictions_json_path)
    if ($if_succ){

        echo '成功上传文件';
        // echo $ret;
        // return json_encode($ret,JSON_UNESCAPED_UNICODE);
    }
    else{
        echo '文件上传失败' ;
    }


}
else{
    echo "file is empty!!!";
}


?>