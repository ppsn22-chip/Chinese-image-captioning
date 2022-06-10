package nelson.takepicture;

import android.app.Activity;
import android.content.Context;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import com.iflytek.cloud.ErrorCode;
import com.iflytek.cloud.InitListener;
import com.iflytek.cloud.SpeechConstant;
import com.iflytek.cloud.SpeechSynthesizer;

public class SpeakerUtil {
    // 语音合成对象
    private static SpeechSynthesizer speechSynthesizer;

    // 语记安装助手类
//    private static ApkInstaller apkInstaller;

    /**
     *
     * 引擎类型:
     * 1. 如果是SpeechConstant.TYPE_CLOUD,代表是采用云端语音合成
     * 2. 如果是SpeechConstant.TYPE_LOCAL,代表是采用本地语音合成，需要下载一个讯飞语音助手
     *
     */
    private static final String mEngineType = SpeechConstant.TYPE_CLOUD;

    //发音人
    private static final String voiceName = "xiaoyan";

    /**
     * 初始化工作
     *
     * @param context the context
     */
    public static void init(Context context) {
        if (speechSynthesizer != null) {
            speechSynthesizer.resumeSpeaking();
            return;
        }

        // 初始化合成对象
        speechSynthesizer = SpeechSynthesizer.createSynthesizer(context,
                mTtsInitListener);
    }

    /**
     * 调用此函数，合成语音
     *
     * @param activity the activity
     * @param text     the text
     */
    public static void startSpeaking(Activity activity, String text) {

        init(activity);
        System.out.println("完成初始化");

        int code = speechSynthesizer.startSpeaking(text, null);
        System.out.println("语音合成返回状态"+ code);
        if (code != ErrorCode.SUCCESS) {
            if (code == ErrorCode.ERROR_COMPONENT_NOT_INSTALLED) {
                // 未安装则跳转到提示安装页面
//                if (apkInstaller == null) {
//                    apkInstaller = new ApkInstaller();
//                }
//                apkInstaller.install(activity);
                Toast.makeText(activity, "您需要下载语音助手", Toast.LENGTH_LONG).show();
            } else {
                Log.e("SpeechSynthesizer", "======语音合成失败 code=" + code);
            }
        }
    }

    /**
     *
     * 初始化回调监听
     *
     **/
    private static InitListener mTtsInitListener = new InitListener() {
        @Override
        public void onInit(int code) {
            if (code != ErrorCode.SUCCESS) {
                Log.e("SpeechSynthesizer", "======初始化失败,错误码 code=" + code);
            } else {
                setParam();
            }
        }
    };

    /**
     *
     * 设置云端参数
     *
     * */
    private static void setParam(){
        // 清空参数
        speechSynthesizer.setParameter(SpeechConstant.PARAMS, null);
        // 根据合成引擎设置相应参数
        if(mEngineType.equals(SpeechConstant.TYPE_CLOUD)) {
            speechSynthesizer.setParameter(SpeechConstant.ENGINE_TYPE, SpeechConstant.TYPE_CLOUD);
            // 设置在线合成发音人
            speechSynthesizer.setParameter(SpeechConstant.VOICE_NAME, voiceName);
            //设置合成语速
            speechSynthesizer.setParameter(SpeechConstant.SPEED, "30");//mSharedPreferences.getString("speed_preference", "50")
            //设置合成音调
            speechSynthesizer.setParameter(SpeechConstant.PITCH, "50");//mSharedPreferences.getString("pitch_preference", "50")
            //设置合成音量
            speechSynthesizer.setParameter(SpeechConstant.VOLUME, "50");//mSharedPreferences.getString("volume_preference", "50")
        }else {
            speechSynthesizer.setParameter(SpeechConstant.ENGINE_TYPE, SpeechConstant.TYPE_LOCAL);
            // 设置本地合成发音人 voicer为空，默认通过语记界面指定发音人。
            speechSynthesizer.setParameter(SpeechConstant.VOICE_NAME, "");

            //本地合成不设置语速、音调、音量，默认使用语记设置,开发者如需自定义参数，请参考在线合成参数设置
        }
        //设置播放器音频流类型
        speechSynthesizer.setParameter(SpeechConstant.STREAM_TYPE,"3");// mSharedPreferences.getString("stream_preference", "3")
        // 设置播放合成音频打断音乐播放，默认为true
        speechSynthesizer.setParameter(SpeechConstant.KEY_REQUEST_FOCUS, "true");

        // 设置音频保存路径，保存音频格式支持pcm、wav，设置路径为sd卡请注意WRITE_EXTERNAL_STORAGE权限
        // 注：AUDIO_FORMAT参数语记需要更新版本才能生效
        speechSynthesizer.setParameter(SpeechConstant.AUDIO_FORMAT, "wav");
        speechSynthesizer.setParameter(SpeechConstant.TTS_AUDIO_PATH, Environment.getExternalStorageDirectory()+"/msc/tts.wav");
    }

    /**
     * 销毁掉第一个
     */
    public static void onDestroy() {
        if (speechSynthesizer == null)
            return;

        speechSynthesizer.stopSpeaking();
        // 退出时释放连接
        speechSynthesizer.destroy();
    }
}
