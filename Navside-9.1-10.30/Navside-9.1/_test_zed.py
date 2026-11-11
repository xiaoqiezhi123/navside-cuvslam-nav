from perception.zed_depth_perception import ZedDepthPerception
import numpy as np
z = ZedDepthPerception()
z.start()
for i in range(10):
    out = z.read()
    nan_count = np.isnan(out.depth_input).sum() if out.depth_input is not None else -1
    print(f'frame {i}: success={out.success} nan_count={nan_count} front_dist={out.front_distance_m:.3f}')
z.close()
