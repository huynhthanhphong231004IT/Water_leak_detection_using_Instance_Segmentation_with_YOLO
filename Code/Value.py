class Value:
    PATH_MODEL = "Model/LeakWater_Model.pt"
    PATH_INPUT_VIDEO = "Video/VD01.mp4"
    PATH_OUTPUT_VIDEO = "Video/Predict_output_video/Predict_output_VD01.mp4"

    PATH_INPUT_IMAGE = "Images/Images_01.jpg"
    PATH_OUTPUT_IMAGE = "Images/Predict_output_img/Predict_output_img_01.jpg"

    Spillway_crest = 0.001
    pixel_to_m2_ratio = 0.0001
    flow_rate_coef = 0.005