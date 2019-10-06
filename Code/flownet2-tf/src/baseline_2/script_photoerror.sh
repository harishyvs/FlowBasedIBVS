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
python example_me.py --width 128 --height 128  --scene /home/yvsharish/try/habitat-sim/data_1/gibson/Kerrtown.glb  --max_frames 20 --save_png --depth_sensor
# conda deactivate

cd /home/yvsharish/ICRA/Flow_Net/flownet2/
source set-env.sh
cd scripts
################## modify here when you change the destination and for iterations
python run-flownet.py /home/yvsharish/ICRA/Flow_Net/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 /home/yvsharish/ICRA/Flow_Net/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template /home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00000.png /home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00019.png out_single_1.flo

cd /home/yvsharish/working/baseline_2/
python ibvs_controller_single.py 0

# conda activate habitat_e
cd /home/yvsharish/working/habitat-sim/examples_baseline_2/
python example_me.py --width 128 --height 128  --scene /home/yvsharish/try/habitat-sim/data_1/gibson/Kerrtown.glb  --max_frames 20 --total_frames 1 --save_png --depth_sensor
# conda deactivate

total_frames=1

cd /home/yvsharish/working/baseline_2/
##################### change here when you change for iterations
python photo_error.py $total_frames 20

filename='/scratch/yvsharish/working/aaaaa/baseline_2_photo.txt'
# n=1
while read line; do
# reading each line
echo "$line"
n=$line
# num=$line
done < $filename

total_frames=1
# while  [ $(echo "$n > 0.05" | bc) -eq 1 ]
while  [[ $(echo "$n > 1000" | bc) -eq 1  ]]
do
          cd /home/yvsharish/ICRA/Flow_Net/flownet2/
          source set-env.sh
          cd scripts
          #################chage here when you chage the iterations
          python run-flownet_baseline_2.py /home/yvsharish/ICRA/Flow_Net/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5 /home/yvsharish/ICRA/Flow_Net/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template $total_frames /home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba.00019.png out_single_1.flo

          cd /home/yvsharish/working/baseline_2/
          python ibvs_controller_single.py $total_frames

          total_frames=$(( $total_frames + 1 ))
          # conda activate habitat_e
          cd /home/yvsharish/working//habitat-sim/examples_baseline_2/
                    python example_me.py --width 128 --height 128  --scene /home/yvsharish/try/habitat-sim/data_1/gibson/Kerrtown.glb  --max_frames 20 --total_frames $total_frames --save_png --depth_sensor
          # conda deactivate

          # harsha=$(($total_frames-1 ))
          # echo "$harsha"
          cd /home/yvsharish/working/baseline_2/
          ############change herer for the number of iterations
          python  photo_error.py  $total_frames  20


          filename='/scratch/yvsharish/working/aaaaa/baseline_2_photo.txt'
          # n=1
          while read line; do
          # reading each line
          echo "$line"
          n=$line
          # num=$line
          done < $filename

          # total_frames=$(( $total_frames + 1 ))

done
