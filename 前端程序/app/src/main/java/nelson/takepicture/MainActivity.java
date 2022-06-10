package nelson.takepicture;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.alibaba.fastjson.JSONObject;
import com.iflytek.cloud.SpeechConstant;
import com.iflytek.cloud.SpeechUtility;

import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

//import org.json.JSONObject;

public class MainActivity extends AppCompatActivity implements OnClickListener  {
    private Button mTakePhoto;
    private ImageView mImageView;
    private TextView captionTextview;
    private static final String TAG = "upload";

    private Dialog mWeiboDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTakePhoto = (Button) findViewById(R.id.takePhotoBtn);
        mImageView = (ImageView) findViewById(R.id.mImageView);
        captionTextview = (TextView) findViewById(R.id.captionTextview);

//        初始化科大讯飞的在线语音合成
        SpeechUtility.createUtility(this, SpeechConstant.APPID +"=5fadf072");
        SpeakerUtil.init(this);
        SpeakerUtil.startSpeaking(MainActivity.this, "欢迎您使用听我说");
        mTakePhoto.setOnClickListener(this);
    }



    @Override
    public void onClick(View v) {

        int id = v.getId();

        if (id == R.id.takePhotoBtn){
            System.out.println("侦听到点击相机~");
            takePhoto();
        }else{
            System.out.println("Error:NOT a CAMERA IS CALLED!");
        }

    }


    String mCurrentPhotoPath;
    static final int REQUEST_TAKE_PHOTO = 1;
    File photoFile = null;

    private void callCamera(){
        //        获取相机
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        System.out.println("相机意向建立");
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {

//                add by jason  https://www.jianshu.com/p/55eae30d133c
            Uri contentUri =  FileProvider.getUriForFile(this, "nelson.takepicture.fileprovider", photoFile);
            //                Log.i(TAG,"contentUri:" + contentUri);
//
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, contentUri);
            takePictureIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
//                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT,
//                        Uri.fromFile(photoFile));
            // Android 7.0强制启用了被称作 StrictMode的策略，带来的影响就是你的App对外无法暴露file://类型的URI了。

            startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
            System.out.println("after start activity result to take picture");

        }
    }
    private void takePhoto() {
        photoFile = createImageFile();
        System.out.println("执行了创建文件代码和申请权限,但并不是一定成功");
        if (photoFile != null){
            callCamera();
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//        调用相机结束

        if (requestCode == REQUEST_TAKE_PHOTO && resultCode == Activity.RESULT_OK) {
            Log.i(TAG,"拍照结束："+ requestCode);
            mTakePhoto.setEnabled(false);
            mTakePhoto.setTextColor(0xFFD0EFC6);
            setPic();
        }
    }

    private void sendPhoto(Bitmap bitmap) {
//        展示等待画面
        mWeiboDialog = WeiboDialogUtils.createLoadingDialog(MainActivity.this, "解析中...");
        new UploadTask().execute(bitmap);
    }

//    图像上传功能

    @SuppressLint("StaticFieldLeak")
    private class UploadTask extends AsyncTask<Bitmap, Void, Void> {

        String ECHO_code = null;
        JSONObject result_json = null;

        @Override
//        该方法运行在后台线程中。这里将主要负责执行那些很耗时的后台处理工作。可以调用 publishProgress方法来更新实时的任务进度。该方法是抽象方法，子类必须实现。
        protected Void doInBackground(Bitmap... bitmaps) {
            System.out.println("后台执行...");
            if (bitmaps[0] == null)
                return null;
            setProgress(0);

            Bitmap bitmap = bitmaps[0];
            System.out.println("后台上传压缩前" + bitmap.getByteCount()/1024 + "KB");
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream); // convert Bitmap to ByteArrayOutputStream
            InputStream in = new ByteArrayInputStream(stream.toByteArray()); // convert ByteArrayOutputStream to ByteArrayInputStream
            try {
                System.out.println("inputstream 大小(压缩后): "+ in.available()/1024 + "KB");
            } catch (IOException e) {
                e.printStackTrace();
            }
            DefaultHttpClient httpclient = new DefaultHttpClient();
            try {

//                http://ubuntu.mahc.host/SeeingServer/saveImage.php
                HttpPost httppost = new HttpPost("http://36.111.131.37/SeeingServer/saveImage.php"); // server

                MultipartEntity reqEntity = new MultipartEntity();

                reqEntity.addPart("myFile",
                        System.currentTimeMillis() + ".jpg", in);
                httppost.setEntity(reqEntity);
                System.out.println("创建post请求并装载好打包数据");

                Log.i(TAG, "request " + httppost.getRequestLine());
                HttpResponse response = null;
//                尝试post过去并得到返回信息
                try {
                    System.out.println("即将发送执行");
                    response = httpclient.execute(httppost);
                } catch (ClientProtocolException e) {

                    e.printStackTrace();
                    Log.i(TAG,"Error1: "+e);
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.i(TAG, "Error2: " + e);
                }

                if (response != null){
                    Log.i(TAG, "response " + response.getStatusLine().toString());
                    Log.i(TAG,"pure response :"+ response);
                    if(response.getStatusLine().getStatusCode()==200){
                        System.out.println("访问成功");
                    }

//                    ECHO_code = EntityUtils.toString(response.getEntity());
                    ECHO_code = EntityUtils.toString(response.getEntity(),"utf-8");

//                    result = JSONObject.fromObject(ECHO_code);
                    Log.i(TAG,"所有返回信息:"+ ECHO_code);
                    result_json = JSONObject.parseObject(ECHO_code);
                    System.out.println("Json object:" + result_json.getString("Caption"));
                }

            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                System.out.println("访问php页面结束");
            }

            // 关闭 in
            try {
                in.close();
            } catch (IOException e) {

                e.printStackTrace();
            }

            // 关闭 stream
            try {
                stream.close();
            } catch (IOException e) {

                e.printStackTrace();
            }

            return null;
        }

//    在publishProgress方法被调用后，UI 线程将调用这个方法从而在界面上展示任务的进展情况，例如通过一个进度条进行展示。
        @Override
        protected void onProgressUpdate(Void... values) {
            System.out.println("onProgressUpdate");
            super.onProgressUpdate(values);
        }
//在doInBackground 执行完成后，onPostExecute 方法将被UI 线程调用，后台的计算结果将通过该方法传递到UI 线程，并且在界面上展示给用户.
        @Override
        protected void onPostExecute(Void result) {

            WeiboDialogUtils.closeDialog(mWeiboDialog);
            System.out.println("后台执行结束");
//            System.out.println("result :"+ result);
            super.onPostExecute(result);
            Toast.makeText(MainActivity.this, result_json.getString("info"), Toast.LENGTH_LONG).show();

            mTakePhoto.setEnabled(true);
            mTakePhoto.setTextColor(0xffffffff);
            SpeakerUtil.startSpeaking(MainActivity.this, result_json.getString("Caption"));

        }
    }

    @Override
    protected void onResume() {
        System.out.println("准备继续");
        super.onResume();
        Log.i(TAG, "onResume: " + this);
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        Log.i(TAG, "onSaveInstanceState");
    }
//    File photoFile = null;


    /**
     * http://developer.android.com/training/camera/photobasics.html
     */
//    申请权限回调函数
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults){
        switch (requestCode){
            case 1:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    Toast.makeText(this,"读权限获取成功",Toast.LENGTH_SHORT).show();
                }else{
                    Toast.makeText(this, "没有读取权限！\n请接收权限申请或前往设置添加权限！", Toast.LENGTH_SHORT).show();
                }
                break;

            case 2:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    Toast.makeText(this,"写权限获取成功",Toast.LENGTH_SHORT).show();

                    createFolders();
                }else{
                    Toast.makeText(this, "没有写存储权限！\n请接收权限申请或前往设置添加权限！", Toast.LENGTH_SHORT).show();
                }
                break;

            default:
        }
    }

    private void createFolders(){
        System.out.println("创建文件夹");
        System.out.print("查看挂载状态");
        System.out.println(Environment.getExternalStorageState());

        // 已经获取权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

            String storageDir = Environment.getExternalStorageDirectory() + "/picupload";
            File dir = new File(storageDir);
            System.out.println("已获取文件写权限");

            if (!dir.exists())
                if(dir.mkdirs()){
                    System.out.println("文件夹创建成功");
                }else {
                    System.out.println("文件夹创建失败");
                }

        } else if(ActivityCompat.shouldShowRequestPermissionRationale(this,Manifest.permission.WRITE_EXTERNAL_STORAGE)){
            Toast.makeText(this, "您拒绝了文件写权限将无法拍照！\n请选择同意读写磁盘", Toast.LENGTH_SHORT).show();
            // 再次申请权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 2);
        }else{
            System.out.println("向用户申请权限");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 2);
            // ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

    }

    private File createImageFile() {
        System.out.println("在执行文件夹下创建一个图像文件");

        //指定图像路径
        String storageDir = Environment.getExternalStorageDirectory() + "/picupload";
        File dir = new File(storageDir);
        //若文件夹不存在就创建
        if (!dir.exists()){
            createFolders();
        }
        // 如果创建成功或者原本就存在
        if (dir.exists()){
            // 生成图像名称
            @SuppressLint("SimpleDateFormat") String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String imageFileName = "JPEG_" + timeStamp + "_"+".jpg";
            File image = new File(storageDir + "/" + imageFileName);
            // Save a file: path for use with ACTION_VIEW intents
            mCurrentPhotoPath = image.getAbsolutePath();
            System.out.println("图像本地存储地址:"+mCurrentPhotoPath);
            return image;
        }
        else{
            return null;
        }

    }

    private void setPic() {
        System.out.println("设置图像");
        // 每次设置图像前把之前的文本清空
        captionTextview.setText("");
        // Get the dimensions of the View
        int targetW = mImageView.getWidth();
        int targetH = mImageView.getHeight();

        System.out.println("屏幕显示图像的宽度"+ targetW);
        System.out.println("屏幕显示图像的高度"+ targetH);

        // Get the dimensions of the bitmap
        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
//      如果设置为true,则不会获取图像,不分配内存,只会返回图像的尺寸信息
        bmOptions.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(mCurrentPhotoPath, bmOptions);
        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;

        System.out.println("初始图像的宽度"+ photoW);
        System.out.println("初始图像的高度"+ photoH);
        // Determine how much to scale down the image
//        取最小的缩放比,防止填不满空间而拉升导致虚化
        int scaleFactor = Math.min(photoW/targetW, photoH/targetH);

        System.out.println("照片除以屏幕显示的最小值" + scaleFactor);
//        System.out.println("人工查看手机中是不是存在图像");
        // Decode the image file into a Bitmap sized to fill the View
        bmOptions.inJustDecodeBounds = false;
        if((photoH + photoW)>7000) //&& Math.abs(photoH-photoW)<2000
//            如果图像过大,则至少放缩4倍
            bmOptions.inSampleSize = scaleFactor << 2;
        else{
            bmOptions.inSampleSize = scaleFactor << 1;
        }

//      如果 inPurgeable 设为True的话表示使用BitmapFactory创建的Bitmap用于存储Pixel的内存空间在系统内存不足时可以被回收，在应用需要再次访问Bitmap的Pixel时（如绘制Bitmap或是调用getPixel），系统会再次调用BitmapFactory decoder重新生成Bitmap的Pixel数组.
//        防止出现OOM问题
        bmOptions.inPurgeable = true;

        Bitmap bitmap = BitmapFactory.decodeFile(mCurrentPhotoPath, bmOptions);

        System.out.println("图像放缩后的宽度"+ bmOptions.outWidth);
        System.out.println("图像放缩后的高度"+ bmOptions.outHeight);

        System.out.println("图像进入后台前大小：" + bitmap.getByteCount()/1024 + "KB");

        float degree = PictureUtils.readPictureDegree(mCurrentPhotoPath);
        System.out.println("原图像旋转角度:"+degree);

        Bitmap rotatedBMP = PictureUtils.rotate(degree,bitmap);

        mImageView.setImageBitmap(rotatedBMP);
        System.out.println("显示图像");
//        System.out.println("");

        try {
            System.out.println("跳转上传功能");
            sendPhoto(rotatedBMP);
        } catch (Exception e) {

            e.printStackTrace();
        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        SpeakerUtil.onDestroy();
    }
}
