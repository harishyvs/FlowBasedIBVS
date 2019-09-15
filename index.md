## Deep  Flow  Guided  Image  Based  Visual  Servoing

### Abstract

Existing  deep  learning  based  visual  servoing  approaches  regress  the  relative  camera  pose  between  a  pair  of images  circumventing  the  requirement  for  scene’s  depth  and camera  parameters.  However,  estimation  of  accurate  camera pose  on  diverse  scenes  is  a  non-trivial  problem,  thus  existing deep  learning  based  approaches  require  a  huge  amount  of training  data  and  sometimes  fine-tuning  for  adaptation  to  a novel scene. Furthermore, current approaches do not consider underlying geometry of the scene and rely on direct estimation of camera pose. Thus, inaccuracies in prediction of the camera pose  especially  for  distant  goals  leads  to  a  degradation  in  the servoing  performance.  In  this  paper,  we  propose  a  two-fold solution:  (i)  we  consider  optical  flow  as  our  visual  features, which  are  predicted  using  a  deep  neural  network.  The  flow features provide dense correspondences between an image pair that  leads  to  a  precise  convergence  of  the  servoing  approach. (ii)  We  then  integrate  these  flow  features  with  depth  estimate  provided  by  another  neural  network  using  interaction matrix  similar  to  classical  image  based  visual  servoing.  This geometrical  understanding  provided  by  the  depth  integration increases the robustness of the overall system. We present two paradigms  for  depth  estimation  under  single-view  and  two-view  settings.  Through  10  unseen  photo-realistic  simulation environments  and  a  real  scenario  on  an  aerial  robot,  we  show that  our  approach  generalises  to  novel  scenarios  producing precise   and   robust   servoing   performance   for   6   degrees   of freedom  positioning  task  over  diverse  environments  with  even large  camera  transformations  without  any  requirement  for retraining  or  fine-tuning. 

### Pipeline of the Proposed approach

![Pipeline](https://i.imgur.com/8VOqFsb.png)
### Predictions and Results
|Initial Image| Desired Image|PhotoVS| Saxena et al[4]  | Sensor Depth  | Depth Net  | Flow Depth  |
|:-:|---|---|---|---|---|---|
|  ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/init.png = 20x20) |  ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/des.png =20x20) |![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/ROANE/ferror.png =20x20)   |  ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/ROANE/ferror.png =20x20) | ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/ferror.png =20x20)  |  ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/ROANE/ferror.png =20x20) | ![ROANE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/ROANE/ferror.png =20x20)  |
| ![BALLOU](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/BALLOU/init.png =20x20)  |   |   |   |   |   |   |
|  ![STOKES](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/STOKES/init.png =20x20) |   |   |   |   |   |   |
|   ![MESIC](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/MESIC/init.png =20x20)|   |   |   |   |   |   |
|   ![ARKANSAW](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ARKANSAW/init.png)|   |   |   |   |   |   |
|  ![PABLO](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/PABLO/init.png) |   |   |   |   |   |   |
|  ![EUDORA](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/EUDORA/init.png) |   |   |   |   |   |   |
|  ![QUANTICO](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/QUANTICO/init.png) |   |   |   |   |   |   |
| ![HILLSDALE](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/HILLSDALE/init.png)  |   |   |   |   |   |   |
| ![DENMARK](https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/DENMARK/init.png) |   |   |   |   |   |   |
Error table
### Video Explanation

