initial=[0,0,0];
initial_s=[0,1.5,0];
desired=[1.01613033 -0.11908760 -0.47667655];
desired_s= [0.51881695,  1.25684679, -0.80755818];
resultant=[ 0.53003514,  -0.26419938, -0.82365918];
ans1=desired-desired_s;
resultant_s= [ 0.53003514,  1.23580062, -0.82365918 ];
dis1=norm(initial-desired);
dis1_s=norm(initial_s-desired_s);
dis2=norm(resultant-desired);
dis2_s=norm(resultant_s-desired_s);
er=dis2/dis1;
er_s=dis2_s/dis1_s;