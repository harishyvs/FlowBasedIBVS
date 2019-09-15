## Deep  Flow  Guided  Image  Based  Visual  Servoing

### Abstract

Existing  deep  learning  based  visual  servoing  approaches  regress  the  relative  camera  pose  between  a  pair  of images  circumventing  the  requirement  for  scene’s  depth  and camera  parameters.  However,  estimation  of  accurate  camera pose  on  diverse  scenes  is  a  non-trivial  problem,  thus  existing deep  learning  based  approaches  require  a  huge  amount  of training  data  and  sometimes  fine-tuning  for  adaptation  to  a novel scene. Furthermore, current approaches do not consider underlying geometry of the scene and rely on direct estimation of camera pose. Thus, inaccuracies in prediction of the camera pose  especially  for  distant  goals  leads  to  a  degradation  in  the servoing  performance.  In  this  paper,  we  propose  a  two-fold solution:  (i)  we  consider  optical  flow  as  our  visual  features, which  are  predicted  using  a  deep  neural  network.  The  flow features provide dense correspondences between an image pair that  leads  to  a  precise  convergence  of  the  servoing  approach. (ii)  We  then  integrate  these  flow  features  with  depth  estimate  provided  by  another  neural  network  using  interaction matrix  similar  to  classical  image  based  visual  servoing.  This geometrical  understanding  provided  by  the  depth  integration increases the robustness of the overall system. We present two paradigms  for  depth  estimation  under  single-view  and  two-view  settings.  Through  10  unseen  photo-realistic  simulation environments  and  a  real  scenario  on  an  aerial  robot,  we  show that  our  approach  generalises  to  novel  scenarios  producing precise   and   robust   servoing   performance   for   6   degrees   of freedom  positioning  task  over  diverse  environments  with  even large  camera  transformations  without  any  requirement  for retraining  or  fine-tuning. 

### Pipeline of the Proposed approach

![Pipeline](https://i.imgur.com/8VOqFsb.png)
### Predictions and Results

<style>
        body {
  font-family: Helvetica, arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  padding-top: 10px;
  padding-bottom: 10px;
  background-color: white;
  padding: 30px; }

body > *:first-child {
  margin-top: 0 !important; }
body > *:last-child {
  margin-bottom: 0 !important; }

a {
  color: #4183C4; }
a.absent {
  color: #cc0000; }
a.anchor {
  display: block;
  padding-left: 30px;
  margin-left: -30px;
  cursor: pointer;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0; }

h1, h2, h3, h4, h5, h6 {
  margin: 20px 0 10px;
  padding: 0;
  font-weight: bold;
  -webkit-font-smoothing: antialiased;
  cursor: text;
  position: relative; }

h1:hover a.anchor, h2:hover a.anchor, h3:hover a.anchor, h4:hover a.anchor, h5:hover a.anchor, h6:hover a.anchor {
  background: url("../../images/modules/styleguide/para.png") no-repeat 10px center;
  text-decoration: none; }

h1 tt, h1 code {
  font-size: inherit; }

h2 tt, h2 code {
  font-size: inherit; }

h3 tt, h3 code {
  font-size: inherit; }

h4 tt, h4 code {
  font-size: inherit; }

h5 tt, h5 code {
  font-size: inherit; }

h6 tt, h6 code {
  font-size: inherit; }

h1 {
  font-size: 28px;
  color: black; }

h2 {
  font-size: 24px;
  border-bottom: 1px solid #cccccc;
  color: black; }

h3 {
  font-size: 18px; }

h4 {
  font-size: 16px; }

h5 {
  font-size: 14px; }

h6 {
  color: #777777;
  font-size: 14px; }

p, blockquote, ul, ol, dl, li, table, pre {
  margin: 15px 0; }

hr {
  background: transparent url("../../images/modules/pulls/dirty-shade.png") repeat-x 0 0;
  border: 0 none;
  color: #cccccc;
  height: 4px;
  padding: 0; }

body > h2:first-child {
  margin-top: 0;
  padding-top: 0; }
body > h1:first-child {
  margin-top: 0;
  padding-top: 0; }
  body > h1:first-child + h2 {
    margin-top: 0;
    padding-top: 0; }
body > h3:first-child, body > h4:first-child, body > h5:first-child, body > h6:first-child {
  margin-top: 0;
  padding-top: 0; }

a:first-child h1, a:first-child h2, a:first-child h3, a:first-child h4, a:first-child h5, a:first-child h6 {
  margin-top: 0;
  padding-top: 0; }

h1 p, h2 p, h3 p, h4 p, h5 p, h6 p {
  margin-top: 0; }

li p.first {
  display: inline-block; }

ul, ol {
  padding-left: 30px; }

ul :first-child, ol :first-child {
  margin-top: 0; }

ul :last-child, ol :last-child {
  margin-bottom: 0; }

dl {
  padding: 0; }
  dl dt {
    font-size: 14px;
    font-weight: bold;
    font-style: italic;
    padding: 0;
    margin: 15px 0 5px; }
    dl dt:first-child {
      padding: 0; }
    dl dt > :first-child {
      margin-top: 0; }
    dl dt > :last-child {
      margin-bottom: 0; }
  dl dd {
    margin: 0 0 15px;
    padding: 0 15px; }
    dl dd > :first-child {
      margin-top: 0; }
    dl dd > :last-child {
      margin-bottom: 0; }

blockquote {
  border-left: 4px solid #dddddd;
  padding: 0 15px;
  color: #777777; }
  blockquote > :first-child {
    margin-top: 0; }
  blockquote > :last-child {
    margin-bottom: 0; }

table {
  padding: 0; }
  table tr {
    border-top: 1px solid #cccccc;
    background-color: white;
    margin: 0;
    padding: 0; }
    table tr:nth-child(2n) {
      background-color: #f8f8f8; }
    table tr th {
      font-weight: bold;
      border: 1px solid #cccccc;
      text-align: left;
      margin: 0;
      padding: 6px 13px; }
    table tr td {
      border: 1px solid #cccccc;
      text-align: left;
      margin: 0;
      padding: 6px 13px; }
    table tr th :first-child, table tr td :first-child {
      margin-top: 0; }
    table tr th :last-child, table tr td :last-child {
      margin-bottom: 0; }

img {
  max-width: 100%; }

span.frame {
  display: block;
  overflow: hidden; }
  span.frame > span {
    border: 1px solid #dddddd;
    display: block;
    float: left;
    overflow: hidden;
    margin: 13px 0 0;
    padding: 7px;
    width: auto; }
  span.frame span img {
    display: block;
    float: left; }
  span.frame span span {
    clear: both;
    color: #333333;
    display: block;
    padding: 5px 0 0; }
span.align-center {
  display: block;
  overflow: hidden;
  clear: both; }
  span.align-center > span {
    display: block;
    overflow: hidden;
    margin: 13px auto 0;
    text-align: center; }
  span.align-center span img {
    margin: 0 auto;
    text-align: center; }
span.align-right {
  display: block;
  overflow: hidden;
  clear: both; }
  span.align-right > span {
    display: block;
    overflow: hidden;
    margin: 13px 0 0;
    text-align: right; }
  span.align-right span img {
    margin: 0;
    text-align: right; }
span.float-left {
  display: block;
  margin-right: 13px;
  overflow: hidden;
  float: left; }
  span.float-left span {
    margin: 13px 0 0; }
span.float-right {
  display: block;
  margin-left: 13px;
  overflow: hidden;
  float: right; }
  span.float-right > span {
    display: block;
    overflow: hidden;
    margin: 13px auto 0;
    text-align: right; }

code, tt {
  margin: 0 2px;
  padding: 0 5px;
  white-space: nowrap;
  border: 1px solid #eaeaea;
  background-color: #f8f8f8;
  border-radius: 3px; }

pre code {
  margin: 0;
  padding: 0;
  white-space: pre;
  border: none;
  background: transparent; }

.highlight pre {
  background-color: #f8f8f8;
  border: 1px solid #cccccc;
  font-size: 13px;
  line-height: 19px;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 3px; }

pre {
  background-color: #f8f8f8;
  border: 1px solid #cccccc;
  font-size: 13px;
  line-height: 19px;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 3px; }
  pre code, pre tt {
    background-color: transparent;
    border: none; }

.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>
|Initial Image|Desired Image|PhotoVS| Saxena et al[4]  | Sensor Depth  | Depth Network  | Flow Depth  |
|:-:|---|---|---|---|---|---|
|  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/init.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/ROANE/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/ROANE/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ROANE/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/ROANE/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/ROANE/ferror.png"> |
| <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/BALLOU/init.png"> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/BALLOU/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/BALLOU/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/BALLOU/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/BALLOU/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/BALLOU/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/BALLOU/ferror.png"> |
|  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/STOKES/init.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/STOKES/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/STOKES/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/STOKES/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/STOKES/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/STOKES/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/STOKES/ferror.png"> |
|   <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/MESIC/init.png"> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/MESIC/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/MESIC/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/MESIC/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/MESIC/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/MESIC/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/MESIC/ferror.png"> |
|   <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ARKANSAW/init.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ARKANSAW/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/ARKANSAW/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/ARKANSAW/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/ARKANSAW/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/ARKANSAW/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/ARKANSAW/ferror.png"> |
|  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/PABLO/init.png"> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/PABLO/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/PABLO/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/PABLO/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/PABLO/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/PABLO/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/PABLO/ferror.png"> |
|  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/EUDORA/init.png"> |  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/EUDORA/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/EUDORA/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/EUDORA/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/EUDORA/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/EUDORA/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/EUDORA/ferror.png"> |
|  <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/QUANTICO/init.png"> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/QUANTICO/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/QUANTICO/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/QUANTICO/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/QUANTICO/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/QUANTICO/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/QUANTICO/ferror.png"> |
| <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/HILLSDALE/init.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/HILLSDALE/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/HILLSDALE/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/HILLSDALE/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/HILLSDALE/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/HILLSDALE/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/HILLSDALE/ferror.png"> |
| <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/DENMARK/init.png"> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/DENMARK/des.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/PhotoVS/DENMARK/ferror.png "> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/ICRA17/DENMARK/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/TrueDepth/DENMARK/ferror.png"> | <img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/DepthNetwork/DENMARK/ferror.png "> |<img align="center" width="100" height="80" src="https://raw.githubusercontent.com/harishyvs/FlowBasedIBVS/master/Work/FlowDepth/DENMARK/ferror.png"> |
{: .tablelines}
### Video Explanation
