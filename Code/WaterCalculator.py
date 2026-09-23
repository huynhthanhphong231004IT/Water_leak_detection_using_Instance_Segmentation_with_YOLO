import time
from Code.Value import Value as Val
class WaterLeakCalculator:
    def __init__(self, pixel_to_m2_ratio= Val.pixel_to_m2_ratio, flow_rate_coef=Val.flow_rate_coef):
        self.pixel_to_m2_ratio = pixel_to_m2_ratio
        self.flow_rate_coef = flow_rate_coef
        self.total_volume_m3 = 0.0
        self.last_time = time.time()

    def calculate_volume(self, mask_pixel_area):
        current_time = time.time()
        dt = current_time - self.last_time  
        self.last_time = current_time
        if mask_pixel_area <= 0:
            return self.total_volume_m3, False
        area_m2 = mask_pixel_area * self.pixel_to_m2_ratio
        flow_rate_m3_s = area_m2 * self.flow_rate_coef
        self.total_volume_m3 += flow_rate_m3_s * dt
        is_alert = self.total_volume_m3 > Val.Spillway_crest
        
        return self.total_volume_m3, is_alert