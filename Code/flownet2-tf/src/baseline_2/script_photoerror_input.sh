#!/bin/bash
if [ ! -d /home/yvsharish/working/habitat-sim/image_baseline_2 ]; then
  mkdir -p /home/yvsharish/working/habitat-sim/image_baseline_2;
fi
if [ ! -d /home/yvsharish/working/habitat-sim/image_baseline_2_output ]; then
  mkdir -p /home/yvsharish/working/habitat-sim/image_baseline_2_output;
fi
# source /home/harish/anaconda2/etc/profile.d/conda.sh

# conda activate habitat_e
cd /home/yvsharish/working/habitat-sim/examples/
###################### modify here when you change the destination and number of iterations
python example_me.py --width 512 --height 384  --scene /home/yvsharish/try/habitat-sim/data_1/gibson/Mesic.glb --sensor_height 1.5 --max_frames 20 --save_png --depth_sensor
# conda deactivate
