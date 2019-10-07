desired=rgb2gray(imread('/home/yvsharish/working/habitat-sim/image_baseline_2_BALLOU_INPUT/test.rgba.00019.png'));
initial=rgb2gray(imread('/home/yvsharish/working/habitat-sim/image_baseline_2_BALLOU_INPUT/test.rgba.00001.png'));
final=rgb2gray(imread('/home/yvsharish/working/habitat-sim/examples/test.rgba.00000.png'));
%finalflow=(imread('/scratch/yvsharish/working/habitat-sim/image_baseline_2_output_MESIC_flowdepth_exp3/test.flo.04828.png'));
figure('Name','True Depth','NumberTitle','off');
output_error=final-desired+127;
input_error= desired-initial+127;
avg_output_error=mean(final(:)-desired(:))  
avg_input_error=mean(desired(:)-initial(:))
rms_output_error=sqrt(mean((final(:) - desired(:)).^2))
rms_input_error=sqrt(mean((desired (:)- initial(:)).^2))

subplot(1,2,1);
imshow(input_error);
%imshow(final);
title('Input Error')
subplot(1,2,2);
imshow(output_error);
%imshow(finalflow);
title('Output Error')
