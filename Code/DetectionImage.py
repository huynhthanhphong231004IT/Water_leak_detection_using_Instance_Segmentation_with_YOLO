import cv2
import numpy as np
from ultralytics import YOLO
from Code.WaterCalculator import WaterLeakCalculator

class WaterLeak_ImageSegmentation:
    @staticmethod
    def Predict(path_model, path_input, path_output):
        model = YOLO(path_model)

        image = cv2.imread(path_input)
        if image is None:
            return

        height, width = image.shape[:2]
        calculator = WaterLeakCalculator(pixel_to_m2_ratio=0.0001, flow_rate_coef=0.0005)

        results = model.predict(source=image, conf=0.5)
        r = results[0]

        annotated_frame = image.copy()
        total_mask_pixels = 0

        if r.masks is not None:
            masks_data = r.masks.data
            total_mask_pixels = int((masks_data > 0).sum().item())

        volume_m3, is_alert = calculator.calculate_volume(total_mask_pixels)

        theme_color = (0, 0, 255) if is_alert else (255, 140, 0)

        if r.masks is not None and r.boxes is not None:
            masks_np = r.masks.data.cpu().numpy()
            boxes = r.boxes

            for mask, box in zip(masks_np, boxes):
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0

                color_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
                color_overlay[mask_bool] = theme_color
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, color_overlay, 0.4, 0)

                contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, theme_color, 2)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), theme_color, 2)

                cls_id = int(box.cls[0])
                label_name = model.names[cls_id]
                conf = float(box.conf[0])
                box_label = f"{label_name} {conf:.2f}"

                (w_label, h_label), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - h_label - 6), (x1 + w_label + 4, y1), theme_color, -1)
                cv2.putText(annotated_frame, box_label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        status_text = "CRITICAL ALERT (> 1 Lit)" if is_alert else "SYSTEM NORMAL"
        vol_str = f"Volume: {volume_m3:.4f} m3 ({volume_m3 * 1000:.2f} Lit)"

        x, y = 30, 40

        cv2.putText(annotated_frame, status_text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, status_text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, theme_color, 1, cv2.LINE_AA)

        cv2.putText(annotated_frame, vol_str, (x, y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, vol_str, (x, y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, theme_color, 1, cv2.LINE_AA)

        cv2.imwrite(path_output, annotated_frame)
        cv2.imshow("YOLO Image Segmentation: Water Leak", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()