package nelson.takepicture;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.ExifInterface;

public class PictureUtils {
        public static float readPictureDegree(String path) {
                int degree = 0;
                try {
                        ExifInterface exifInterface = new ExifInterface(path);
                        int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
                        switch (orientation) {
                                case ExifInterface.ORIENTATION_ROTATE_90:
                                        degree = 90;
                                        break;
                                case ExifInterface.ORIENTATION_ROTATE_180:
                                        degree = 180;
                                        break;
                                case ExifInterface.ORIENTATION_ROTATE_270:
                                        degree = 270;
                                        break;
                        }
                } catch (Exception e) {
                        e.printStackTrace();
                }
                return degree;
        }

        public static Bitmap rotate(float degree, Bitmap bitmap){

                Matrix mtx = new Matrix();
                mtx.postRotate(degree);
                Bitmap rotatedBMP = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), mtx, true);
                if (null != bitmap && (degree -0)!=0 &&rotatedBMP != bitmap) {
                        System.out.println("回收旧图像");
                        bitmap.recycle();
                }
                return rotatedBMP;
        }
}


