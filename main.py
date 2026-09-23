from Code.Value import Value as Val
from Code.Detection import WaterLeak_InstanceSegmentation
from Code.DetectionImage import WaterLeak_ImageSegmentation
if __name__ == "__main__":
    model_path = Val.PATH_MODEL

    # Dự đoán trên hình ảnh
    img_input_path = Val.PATH_INPUT_IMAGE
    img_output_path = Val.PATH_OUTPUT_IMAGE
    WaterLeak_ImageSegmentation.Predict(model_path, img_input_path, img_output_path)

    # Dự đoán trên video
    video_input_path = Val.PATH_INPUT_VIDEO
    video_output_path = Val.PATH_OUTPUT_VIDEO
    WaterLeak_InstanceSegmentation.Predict(model_path, video_input_path, video_output_path)
