# 3D Machine Learning
In recent years, tremendous amount of progress is being made in the field of 3D Machine Learning, which is an interdisciplinary field that fuses computer vision, computer graphics and machine learning. This repo is derived from my study notes and will be used as a place for triaging new research papers. 

I'll use the following icons to differentiate 3D representations:
* :camera: Multi-view Images
* :space_invader: Volumetric
* :game_die: Point Cloud
* :gem: Polygonal Mesh
* :pill: Primitive-based

## Get Involved
To contribute to this Repo, you may add content through pull requests or open an issue to let me know. 

:star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:<br>
We have also created a Slack workplace for people around the globe to ask questions, share knowledge and facilitate collaborations. Together, I'm sure we can advance this field as a collaborative effort. Join the community with [this link](https://join.slack.com/t/3d-machine-learning/shared_invite/enQtMzUyMTgyNzgwOTgzLWIzY2M3MTQ1ODgwOWEwMGU3MWYxMThhOWQzZGY4OTdhM2VlYTc2N2FmNGVmMzE0MGJlNjg1NjA5OTRhNzlkOWQ).
<br>:star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:

## Table of Contents
- [Courses](#courses)
- [Datasets](#datasets)
  - [3D Models](#3d_models)
  - [3D Scenes](#3d_scenes)
- [3D Pose Estimation](#pose_estimation)
- [Single Object Classification](#single_classification)
- [Multiple Objects Detection](#multiple_detection)
- [Scene/Object Semantic Segmentation](#segmentation)
- [3D Geometry Synthesis/Reconstruction](#3d_synthesis)
  - [Parametric Morphable Model-based methods](#3d_synthesis_model_based)
  - [Part-based Template Learning methods](#3d_synthesis_template_based)
  - [Deep Learning Methods](#3d_synthesis_dl_based)
- [Texture/Material Synthesis](#material_synthesis)
- [Style Transfer](#style_transfer)
- [Scene Synthesis/Reconstruction](#scene_synthesis)
- [Scene Understanding](#scene_understanding)

<a name="courses" />

## Available Courses
[UCSD CSE291-I00: Machine Learning for 3D Data (Winter 2018)](https://cse291-i.github.io/index.html)

[Stanford CS468: Machine Learning for 3D Data (Spring 2017)](http://graphics.stanford.edu/courses/cs468-17-spring/)

[MIT 6.838: Shape Analysis (Spring 2017)](http://groups.csail.mit.edu/gdpgroup/6838_spring_2017.html)

[Princeton COS 526: Advanced Computer Graphics  (Fall 2010)](https://www.cs.princeton.edu/courses/archive/fall10/cos526/syllabus.php)

[Princeton CS597: Geometric Modeling and Analysis (Fall 2003)](https://www.cs.princeton.edu/courses/archive/fall03/cs597D/)

[Geometric Deep Learning](http://geometricdeeplearning.com/)

[Paper Collection for 3D Understanding](https://www.cs.princeton.edu/courses/archive/spring15/cos598A/cos598A.html#Estimating)

<a name="datasets" />

<a name="pose_estimation" />

## Scene/Object Semantic Segmentation
<b>Learning 3D Mesh Segmentation and Labeling (2010)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/LabelMeshes.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/0bf390e2a14f74bcc8838d5fb1c0c4cc60e92eb7/7-Figure7-1.png" /></p>

<b>Unsupervised Co-Segmentation of a Set of Shapes via Descriptor-Space Spectral Clustering (2011)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/sidi_siga11_coseg.pdf)
<p align="center"><img width="30%" src="http://people.scs.carleton.ca/~olivervankaick/cosegmentation/results6.png" /></p>

<b>Single-View Reconstruction via Joint Analysis of Image and Shape Collections (2015)</b> [[Paper]](https://www.cs.utexas.edu/~huangqx/modeling_sig15.pdf) [[Code]](https://github.com/huangqx/image_shape_align)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2015/05/single-view.png" /></p>

<b>3D Shape Segmentation with Projective Convolutional Networks (2017)</b> [[Paper]](http://people.cs.umass.edu/~kalo/papers/shapepfcn/) [[Code]](https://github.com/kalov/ShapePFCN)
<p align="center"><img width="50%" src="http://people.cs.umass.edu/~kalo/papers/shapepfcn/teaser.jpg" /></p>

<b>Learning Hierarchical Shape Segmentation and Labeling from Online Repositories (2017)</b> [[Paper]](http://cs.stanford.edu/~ericyi/project_page/hier_seg/index.html)
<p align="center"><img width="50%" src="http://cs.stanford.edu/~ericyi/project_page/hier_seg/figures/teaser.jpg" /></p>

:space_invader: <b>ScanNet (2017)</b> [[Paper]](https://arxiv.org/pdf/1702.04405.pdf) [[Code]](https://github.com/scannet/scannet)
<p align="center"><img width="50%" src="http://www.scan-net.org/img/voxel-predictions.jpg" /></p>

:game_die: <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/) [[Code]](https://github.com/charlesq34/pointnet)
<p align="center"><img width="40%" src="https://web.stanford.edu/~rqi/papers/pointnet.png" /></p>

:game_die: <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf) [[Code]](https://github.com/charlesq34/pointnet2)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointNet%2B%2B-%20Deep%20Hierarchical%20Feature%20Learning%20on%20Point%20Sets%20in%20a%20Metric%20Space.png" /></p>

:game_die: <b>3D Graph Neural Networks for RGBD Semantic Segmentation (2017)</b> [[Paper]](http://www.cs.toronto.edu/~rjliao/papers/iccv_2017_3DGNN.pdf)
<p align="center"><img width="40%" src="http://www.fonow.com/Images/2017-10-18/66372-20171018115809740-2125227250.jpg" /></p>

:game_die: <b>3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic
Parsing of Large-scale 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.06783.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/3DCNN-DQN-RNN.png" /></p>

:game_die::space_invader: <b>Semantic Segmentation of Indoor Point Clouds using Convolutional Neural Networks (2017)</b> [[Paper]](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-4-W4/101/2017/isprs-annals-IV-4-W4-101-2017.pdf)
<p align="center"><img width="55%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Semantic Segmentation of Indoor Point Clouds using Convolutional Neural Networks.png" /></p>

:game_die::space_invader: <b>SEGCloud: Semantic Segmentation of 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.07563.pdf)
<p align="center"><img width="55%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/SEGCloud.png" /></p>

:game_die::space_invader: <b>Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55 (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.06104.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Core55.png" /></p>

:game_die: <b>Dynamic Graph CNN for Learning on Point Clouds (2018)</b> [[Paper]](https://arxiv.org/pdf/1801.07829.pdf)
<p align="center"><img width="50%" src="https://liuziwei7.github.io/homepage_files/dynamicgcnn_logo.png" /></p>

:game_die: <b>PointCNN (2018)</b> [[Paper]](https://yangyanli.github.io/PointCNN/)
<p align="center"><img width="50%" src="http://yangyan.li/images/paper/pointcnn.png" /></p>

:camera::space_invader: <b>3DMV: Joint 3D-Multi-View Prediction for 3D Semantic Scene Segmentation (2018)</b> [[Paper]](https://arxiv.org/pdf/1803.10409.pdf)
<p align="center"><img width="50%" src="https://cs.stanford.edu/~adai/papers/2018/1threedmv/teaser.jpg" /></p>

:space_invader: <b>ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.10215.pdf) 
<p align="center"><img width="50%" src="https://cs.stanford.edu/~adai/papers/2018/0scancomplete/teaser.jpg" /></p>

:game_die::camera: <b>SPLATNet: Sparse Lattice Networks for Point Cloud Processing (2018)</b> [[Paper]](https://arxiv.org/pdf/1802.08275.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/SPLATNet-%20Sparse%20Lattice%20Networks%20for%20Point%20Cloud%20Processing.jpeg" /></p>

<a name="3d_synthesis" />

## 3D Model Synthesis/Reconstruction

<a name="3d_synthesis_model_based" />

_Parametric Morphable Model-based methods_

<b>A Morphable Model For The Synthesis Of 3D Faces (1999)</b> [[Paper]](http://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf)[[Code]](https://github.com/MichaelMure/3DMM)
<p align="center"><img width="40%" src="http://mblogthumb3.phinf.naver.net/MjAxNzAzMTdfMjcz/MDAxNDg5NzE3MzU0ODI3.9lQioLxwoGmtoIVXX9sbVOzhezoqgKMKiTovBnbUFN0g.sXN5tG4Kohgk7OJEtPnux-mv7OAoXVxxCyo3SGZMc6Yg.PNG.atelierjpro/031717_0222_DataDrivenS4.png?type=w420" /></p>

<b>The Space of Human Body Shapes: Reconstruction and Parameterization from Range Scans (2003)</b> [[Paper]](http://grail.cs.washington.edu/projects/digital-human/pub/allen03space-submit.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/46d39b0e21ae956e4bcb7a789f92be480d45ee12/7-Figure10-1.png" /></p>

<b>Category-Specific Object Reconstruction from a Single Image (2014)</b> [[Paper]](https://people.eecs.berkeley.edu/~akar/categoryshapes.pdf)
<p align="center"><img width="50%" src="http://people.eecs.berkeley.edu/~akar/categoryShapes/images/teaser.png" /></p>

:game_die: <b>DeformNet: Free-Form Deformation Network for 3D Shape Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf)
<p align="center"><img width="50%" src="https://chrischoy.github.io/images/publication/deformnet/model.png" /></p>

:gem: <b>Mesh-based Autoencoders for Localized Deformation Component Analysis (2017)</b> [[Paper]](https://arxiv.org/pdf/1709.04304.pdf)
<p align="center"><img width="50%" src="http://qytan.com/img/point_conv.jpg" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>

:gem: <b>Using Locally Corresponding CAD Models for
Dense 3D Reconstructions from a Single Image (2017)</b> [[Paper]](http://ci2cv.net/media/papers/chenkong_cvpr_2017.pdf)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r02.png" /></p>

:gem: <b>Compact Model Representation for 3D Reconstruction (2017)</b> [[Paper]](https://jhonykaesemodel.com/publication/3dv2017/)
<p align="center"><img width="50%" src="https://jhonykaesemodel.com/img/headers/overview.png" /></p>

:gem: <b>Image2Mesh: A Learning Framework for Single Image 3D Reconstruction (2017)</b> [[Paper]](https://arxiv.org/pdf/1711.10669.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DW5VhjpW4AAESHO.jpg" /></p>

:gem: <b>Learning free-form deformations for 3D object reconstruction (2018)</b> [[Paper]](https://jhonykaesemodel.com/publication/learning_ffd/)
<p align="center"><img width="50%" src="https://jhonykaesemodel.com/learning_ffd_overview.png" /></p>

:gem: <b>Variational Autoencoders for Deforming 3D Mesh Models(2018 CVPR)</b> [[Paper]](http://qytan.com/publication/vae/)
<p align="center"><img width="50%" src="http://humanmotion.ict.ac.cn/papers/2018P5_VariationalAutoencoders/TeaserImage.jpg" /></p>

:gem: <b>Lions and Tigers and Bears: Capturing Non-Rigid, 3D, Articulated Shape from Images (2018 CVPR)</b> [[Paper]](http://files.is.tue.mpg.de/black/papers/zuffiCVPR2018.pdf)
<p align="center"><img width="50%" src="https://3c1703fe8d.site.internapcdn.net/newman/gfx/news/hires/2018/realisticava.jpg" /></p>

<a name="3d_synthesis_template_based" />

_Part-based Template Learning methods_

<b>Modeling by Example (2004)</b> [[Paper]](http://www.cs.princeton.edu/~funk/sig04a.pdf)
<p align="center"><img width="20%" src="http://gfx.cs.princeton.edu/pubs/Funkhouser_2004_MBE/chair.jpg" /></p>

<b>Model Composition from Interchangeable Components (2007)</b> [[Paper]](http://www.cs.princeton.edu/courses/archive/spring11/cos598A/pdfs/Kraevoy07.pdf)
<p align="center"><img width="40%" src="http://www.cs.ubc.ca/labs/imager/tr/2007/Vlad_Shuffler/teaser.jpg" /></p>

<b>Data-Driven Suggestions for Creativity Support in 3D Modeling (2010)</b> [[Paper]](http://vladlen.info/publications/data-driven-suggestions-for-creativity-support-in-3d-modeling/)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/creativity.png" /></p>

<b>Photo-Inspired Model-Driven 3D Object Modeling (2011)</b> [[Paper]](http://kevinkaixu.net/projects/photo-inspired.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/photo-inspired/overview.PNG" /></p>

<b>Probabilistic Reasoning for Assembly-Based 3D Modeling (2011)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/assembly/ProbReasoningShapeModeling.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/highlight9.png" /></p>

<b>A Probabilistic Model for Component-Based Shape Synthesis (2012)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/ShapeSynthesis/ShapeSynthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/A%20Probabilistic%20Model%20for%20Component-Based%20Shape%20Synthesis.png" /></p>

<b>Structure Recovery by Part Assembly (2012)</b> [[Paper]](http://cg.cs.tsinghua.edu.cn/StructureRecovery/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Structure%20Recovery%20by%20Part%20Assembly.png" /></p>

<b>Fit and Diverse: Set Evolution for Inspiring 3D Shape Galleries (2012)</b> [[Paper]](http://kevinkaixu.net/projects/civil.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/civil/teaser.png" /></p>

<b>AttribIt: Content Creation with Semantic Attributes (2013)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/attribit/AttribIt.pdf)
<p align="center"><img width="30%" src="http://gfx.cs.princeton.edu/gfx/pubs/Chaudhuri_2013_ACC/teaser.jpg" /></p>

<b>Learning Part-based Templates from Large Collections of 3D Shapes (2013)</b> [[Paper]](http://shape.cs.princeton.edu/vkcorrs/papers/13_SIGGRAPH_CorrsTmplt.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Learning%20Part-based%20Templates%20from%20Large%20Collections%20of%203D%20Shapes.png" /></p>

<b>Topology-Varying 3D Shape Creation via Structural Blending (2014)</b> [[Paper]](http://gruvi.cs.sfu.ca/project/topo/)
<p align="center"><img width="50%" src="https://i.ytimg.com/vi/Xc4qf7v6a-w/maxresdefault.jpg" /></p>

<b>Estimating Image Depth using Shape Collections (2014)</b> [[Paper]](http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/imageShapeNet_sigg14.html)
<p align="center"><img width="50%" src="http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/paper_docs/pipeline.jpg" /></p>

<b>Single-View Reconstruction via Joint Analysis of Image and Shape Collections (2015)</b> [[Paper]](https://www.cs.utexas.edu/~huangqx/modeling_sig15.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2015/05/single-view.png" /></p>

<b>Interchangeable Components for Hands-On Assembly Based Modeling (2016)</b> [[Paper]](http://www.cs.umb.edu/~craigyu/papers/handson_low_res.pdf)
<p align="center"><img width="30%" src="https://github.com/timzhang642/test1/blob/master/imgs/Interchangeable%20Components%20for%20Hands-On%20Assembly%20Based%20Modeling.png" /></p>

<b>Shape Completion from a Single RGBD Image (2016)</b> [[Paper]](http://www.kunzhou.net/2016/shapecompletion-tvcg16.pdf)
<p align="center"><img width="40%" src="http://tianjiashao.com/Images/2015/completion.jpg" /></p>

<a name="3d_synthesis_dl_based" />

_Deep Learning Methods_

:camera: <b>Learning to Generate Chairs, Tables and Cars with Convolutional Networks (2014)</b> [[Paper]](https://arxiv.org/pdf/1411.5928.pdf)
<p align="center"><img width="50%" src="https://zo7.github.io/img/2016-09-25-generating-faces/chairs-model.png" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015, NIPS)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/jimeiyang/deepRotator/blob/master/demo_img.png" /></p>

:game_die: <b>Analysis and synthesis of 3D shape families via deep-learned generative models of surfaces (2015)</b> [[Paper]](https://people.cs.umass.edu/~hbhuang/publications/bsm/)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~hbhuang/publications/bsm/bsm_teaser.jpg" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf) [[Code]](https://github.com/jimeiyang/deepRotator)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/042993c46294a542946c9c1706b7b22deb1d7c43/2-Figure1-1.png" /></p>

:camera: <b>Multi-view 3D Models from Single Images with a Convolutional Network (2016)</b> [[Paper]](https://arxiv.org/pdf/1511.06702.pdf) [[Code]](https://github.com/lmb-freiburg/mv3d)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/3d7ca5ad34f23a5fab16e73e287d1a059dc7ef9a/4-Figure2-1.png" /></p>

:camera: <b>View Synthesis by Appearance Flow (2016)</b> [[Paper]](https://people.eecs.berkeley.edu/~tinghuiz/papers/eccv16_appflow.pdf) [[Code]](https://github.com/tinghuiz/appearance-flow)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/12280506dc8b5c3ca2db29fc3be694d9a8bef48c/6-Figure2-1.png" /></p>

:space_invader: <b>Voxlets: Structured Prediction of Unobserved Voxels From a Single Depth Image (2016)</b> [[Paper]](http://visual.cs.ucl.ac.uk/pubs/depthPrediction/http://visual.cs.ucl.ac.uk/pubs/depthPrediction/) [[Code]](https://github.com/mdfirman/voxlets)
<p align="center"><img width="30%" src="https://i.ytimg.com/vi/1wy4y2GWD5o/maxresdefault.jpg" /></p>

:space_invader: <b>3D-R2N2: 3D Recurrent Reconstruction Neural Network (2016)</b> [[Paper]](http://cvgl.stanford.edu/3d-r2n2/) [[Code]](https://github.com/chrischoy/3D-R2N2)
<p align="center"><img width="50%" src="http://3d-r2n2.stanford.edu/imgs/overview.png" /></p>

:space_invader: <b>Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision (2016)</b> [[Paper]](https://eng.ucmerced.edu/people/jyang44/papers/nips16_ptn.pdf)
<p align="center"><img width="70%" src="https://sites.google.com/site/skywalkeryxc/_/rsrc/1481104596238/perspective_transformer_nets/network_arch.png" /></p>

:space_invader: <b>TL-Embedding Network: Learning a Predictable and Generative Vector Representation for Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1603.08637.pdf)
<p align="center"><img width="50%" src="https://rohitgirdhar.github.io/GenerativePredictableVoxels/assets/webteaser.jpg" /></p>

:space_invader: <b>3D GAN: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (2016)</b> [[Paper]](https://arxiv.org/pdf/1610.07584.pdf)
<p align="center"><img width="50%" src="http://3dgan.csail.mit.edu/images/model.jpg" /></p>

:space_invader: <b>3D Shape Induction from 2D Views of Multiple Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1612.05872.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/e78572eeef8b967dec420013c65a6684487c13b2/2-Figure2-1.png" /></p>

:camera: <b>Unsupervised Learning of 3D Structure from Images (2016)</b> [[Paper]](https://arxiv.org/pdf/1607.00662.pdf)
<p align="center"><img width="50%" src="https://adriancolyer.files.wordpress.com/2016/12/unsupervised-3d-fig-10.jpeg?w=600" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2016)</b> [[Paper]](https://arxiv.org/pdf/1608.04236.pdf) [[Code]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="http://davidstutz.de/wordpress/wp-content/uploads/2017/02/brock_vae.png" /></p>

:camera: <b>Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency (2017)</b> [[Paper]](https://shubhtuls.github.io/drc/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/drc/resources/images/teaserChair.png" /></p>

:camera: <b>Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and Silhouettes with Deep Generative Networks (2017)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Soltani_Synthesizing_3D_Shapes_CVPR_2017_paper.pdf)  [[Code]](https://github.com/Amir-Arsalan/Synthesize3DviaDepthOrSil)
<p align="center"><img width="50%" src="https://jiajunwu.com/images/spotlight_3dvae.jpg" /></p>

:space_invader: <b>Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis (2017)</b> [[Paper]](https://arxiv.org/pdf/1612.00101.pdf) [[Code]](https://github.com/angeladai/cnncomplete)
<p align="center"><img width="50%" src="http://graphics.stanford.edu/projects/cnncomplete/teaser.jpg" /></p>

:space_invader: <b>Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.09438.pdf) [[Code]](https://github.com/lmb-freiburg/ogn)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/6c2a292bb018a8742cbb0bbc5e23dd0a454ffe3a/2-Figure2-1.png" /></p>

:space_invader: <b>Hierarchical Surface Prediction for 3D Object Reconstruction (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.00710.pdf)
<p align="center"><img width="50%" src="http://bair.berkeley.edu/blog/assets/hsp/image_2.png" /></p>

:space_invader: <b>OctNetFusion: Learning Depth Fusion from Data (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.01047.pdf) [[Code]](https://github.com/griegler/octnetfusion)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/OctNetFusion-%20Learning%20Depth%20Fusion%20from%20Data.jpeg" /></p>

:game_die: <b>A Point Set Generation Network for 3D Object Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf) [[Code]](https://github.com/fanhqme/PointSetGeneration)
<p align="center"><img width="50%" src="http://gting.me/2017/07/17/pr-point-set-generation-from-single-image/ps3d_introduction.PNG" /></p>

:game_die: <b>Learning Representations and Generative Models for 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.02392.pdf) [[Code]](https://github.com/optas/latent_3d_points)
<p align="center"><img width="50%" src="https://github.com/optas/latent_3d_points/blob/master/doc/images/teaser.jpg" /></p>

:game_die: <b>Shape Generation using Spatially Partitioned Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.06267.pdf)
<p align="center"><img width="50%" src="http://mgadelha.me/sppc/fig/abstract.png" /></p>

:game_die: <b>PCPNET Learning Local Shape Properties from Raw Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.04954.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PCPNET%20Learning%20Local%20Shape%20Properties%20from%20Raw%20Point%20Clouds%20(2017).jpeg" /></p>

:camera: <b>Transformation-Grounded Image Generation Network for Novel 3D View Synthesis (2017)</b> [[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/) [[Code]](https://github.com/silverbottlep/tvsn)
<p align="center"><img width="50%" src="https://eng.ucmerced.edu/people/jyang44/pics/view_synthesis.gif" /></p>

:camera: <b>Tag Disentangled Generative Adversarial Networks for Object Image Re-rendering (2017)</b> [[Paper]](http://static.ijcai.org/proceedings-2017/0404.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Tag%20Disentangled%20Generative%20Adversarial%20Networks%20for%20Object%20Image%20Re-rendering.jpeg" /></p>

:camera: <b>3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks (2017)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/SketchModeling/) [[Code]](https://github.com/happylun/SketchModeling)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/SketchModeling/SketchModeling_teaser.png" /></p>

:space_invader: <b>Interactive 3D Modeling with a Generative Adversarial Network (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.05170.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DCsPKLqXoAEBd-V.jpg" /></p>

:camera::space_invader: <b>Weakly supervised 3D Reconstruction with Adversarial Constraint (2017)</b> [[Paper]](https://arxiv.org/pdf/1705.10904.pdf) [[Code]](https://github.com/jgwak/McRecon)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Weakly%20supervised%203D%20Reconstruction%20with%20Adversarial%20Constraint%20(2017).jpeg" /></p>

:camera: <b>SurfNet: Generating 3D shape surfaces using deep residual networks (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.04079.pdf)
<p align="center"><img width="50%" src="https://3dadept.com/wp-content/uploads/2017/07/Screenshot-from-2017-07-26-145521-e1501077539723.png" /></p>

:pill: <b>GRASS: Generative Recursive Autoencoders for Shape Structures (SIGGRAPH 2017)</b> [[Paper]](http://kevinkaixu.net/projects/grass.html) [[Code]](https://github.com/junli-lj/grass) [[code]](https://github.com/kevin-kaixu/grass_pytorch)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/grass/teaser.jpg" /></p>

:pill: <b> 3D-PRNN: Generating Shape Primitives with Recurrent Neural Networks (2017)</b> [[Paper]](https://arxiv.org/pdf/1708.01648.pdf)[[code]](https://github.com/zouchuhang/3D-PRNN)
<p align="center"><img width="50%" src="https://github.com/zouchuhang/3D-PRNN/blob/master/figs/teasor.jpg" /></p>

:gem: <b>Neural 3D Mesh Renderer (2017)</b> [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DPSm-4HWkAApEZd.jpg" /></p>

:game_die::space_invader: <b>Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55 (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.06104.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Core55.png" /></p>

:space_invader: <b>Pix2vox: Sketch-Based 3D Exploration with Stacked Generative Adversarial Networks (2017)</b> [[Code]](https://github.com/maxorange/pix2vox)
<p align="center"><img width="50%" src="https://github.com/maxorange/pix2vox/blob/master/img/sample.gif" /></p>

:camera::space_invader: <b>What You Sketch Is What You Get: 3D Sketching using Multi-View Deep Volumetric Prediction (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.08390.pdf)
<p align="center"><img width="50%" src="https://arxiv-sanity-sanity-production.s3.amazonaws.com/render-output/31631/x1.png" /></p>

:camera::space_invader: <b>MarrNet: 3D Shape Reconstruction via 2.5D Sketches (2017)</b> [[Paper]](http://marrnet.csail.mit.edu/)
<p align="center"><img width="50%" src="http://marrnet.csail.mit.edu/images/model.jpg" /></p>

:camera::space_invader::game_die: <b>Learning a Multi-View Stereo Machine (2017 NIPS)</b> [[Paper]](http://bair.berkeley.edu/blog/2017/09/05/unified-3d/) 
<p align="center"><img width="50%" src="http://bair.berkeley.edu/static/blog/unified-3d/Network.png" /></p>

:space_invader: <b>3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions (2017)</b> [[Paper]](http://3dmatch.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://3dmatch.cs.princeton.edu/img/overview.jpg" /></p>

:space_invader: <b>Scaling CNNs for High Resolution Volumetric Reconstruction from a Single Image (2017)</b> [[Paper]](https://ieeexplore.ieee.org/document/8265323/)
<p align="center"><img width="50%" src="https://github.com/frankhjwx/3D-Machine-Learning/blob/master/imgs/Scaling%20CNN%20Reconstruction.png" /></p>

:game_die: <b>PU-Net: Point Cloud Upsampling Network (2018)</b> [[Paper]](https://arxiv.org/pdf/1801.06761.pdf) [[Code]](https://github.com/yulequan/PU-Net)
<p align="center"><img width="50%" src="http://appsrv.cse.cuhk.edu.hk/~lqyu/indexpics/Pu-Net.png" /></p> 

:camera::space_invader: <b>Multi-view Consistency as Supervisory Signal  for Learning Shape and Pose Prediction (2018 CVPR)</b> [[Paper]](https://shubhtuls.github.io/mvcSnP/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/mvcSnP/resources/images/teaser.png" /></p>

:camera::game_die: <b>Object-Centric Photometric Bundle Adjustment with Deep Shape Prior (2018)</b> [[Paper]](http://ci2cv.net/media/papers/WACV18.pdf)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r06.png" /></p>

:camera::game_die: <b>Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction (2018 AAAI)</b> [[Paper]](https://chenhsuanlin.bitbucket.io/3D-point-cloud-generation/)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r05.png" /></p>

:gem: <b>Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (2018)</b> [[Paper]](http://bigvid.fudan.edu.cn/pixel2mesh/)
<p align="center"><img width="50%" src="http://bigvid.fudan.edu.cn/pixel2mesh/eccv2018/pipeline_01.jpg" /></p>

:gem: <b>AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation (2018 CVPR)</b> [[Paper]](http://imagine.enpc.fr/~groueixt/atlasnet/) [[Code]](https://github.com/ThibaultGROUEIX/AtlasNet)
<p align="center"><img width="50%" src="http://imagine.enpc.fr/~groueixt/atlasnet/imgs/teaser.small.png" /></p>

:space_invader::gem: <b>Deep Marching Cubes: Learning Explicit Surface Representations (2018 CVPR)</b> [[Paper]](http://www.cvlibs.net/publications/Liao2018CVPR.pdf)
<p align="center"><img width="50%" src="https://github.com/frankhjwx/3D-Machine-Learning/blob/master/imgs/Deep%20Marching%20Cubes.png" /></p>

:space_invader: <b>Im2Avatar: Colorful 3D Reconstruction from a Single Image (2018)</b> [[Paper]](https://arxiv.org/pdf/1804.06375v1.pdf)
<p align="center"><img width="50%" src="https://arxiv-sanity-sanity-production.s3.amazonaws.com/render-output/113225/figures/teaser.png" /></p>

:gem: <b>Learning Category-Specific Mesh Reconstruction  from Image Collections (2018)</b> [[Paper]](https://akanazawa.github.io/cmr/#)
<p align="center"><img width="50%" src="https://akanazawa.github.io/cmr/resources/images/teaser.png" /></p>

:pill: <b>CSGNet: Neural Shape Parser for Constructive Solid Geometry (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.08290.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DR-RgbaU8AEyjeW.jpg" /></p>

:space_invader: <b>Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings (2018)</b> [[Paper]](http://text2shape.stanford.edu/)
<p align="center"><img width="50%" src="http://text2shape.stanford.edu/figures/pull.png" /></p>

:space_invader::gem::camera: <b>Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation (2018)</b>  [[Paper]](https://arxiv.org/abs/1802.09987) [[Code]](https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation)
<p align="center"><img width="60%" src="imgs/decomposition_new.png" /> <img width="60%" src="imgs/256.png" /></p>

:space_invader::gem::camera: <b>Pixels, voxels, and views: A study of shape representations for single view 3D object shape prediction (2018 CVPR)</b>  [[Paper]](https://arxiv.org/abs/1804.06032)
<p align="center"><img width="60%" src="imgs/pixels-voxels-views-rgb2mesh.png" /> </p>

:camera::game_die: <b>Neural scene representation and rendering (2018)</b> [[Paper]](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
<p align="center"><img width="50%" src="http://www.arimorcos.com/static/images/publication_images/gqn_image.png" /></p>

:pill: <b>Im2Struct: Recovering 3D Shape Structure from a Single RGB Image (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.05469.pdf)
<p align="center"><img width="50%" src="https://kevinkaixu.net/images/publications/niu_cvpr18.jpg" /></p>

:game_die: <b>FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1712.07262.pdf)
<p align="center"><img width="50%" src="http://simbaforrest.github.io/fig/FoldingNet.jpg" /></p>

:camera::space_invader: <b>Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling (2018 CVPR)</b> [[Paper]](http://pix3d.csail.mit.edu/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Pix3D%20-%20Dataset%20and%20Methods%20for%20Single-Image%203D%20Shape%20Modeling%20(2018%20CVPR).png" /></p>

:gem: <b>3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare (2018 CVPR)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1128.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/3D-RCNN-%20Instance-level%203D%20Object%20Reconstruction%20via%20Render-and-Compare%20(2018%20CVPR).jpeg" /></p>

:space_invader: <b>Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.10975.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Matryoshka%20Networks-%20Predicting%203D%20Geometry%20via%20Nested%20Shape%20Layers%20(2018%20CVPR).jpeg" /></p>

<a name="material_synthesis" />

## Texture/Material Synthesis
<b>Texture Synthesis Using Convolutional Neural Networks (2015)</b> [[Paper]](https://arxiv.org/pdf/1505.07376.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Texture%20Synthesis%20Using%20Convolutional%20Neural%20Networks.jpeg" /></p>

<b>Two-Shot SVBRDF Capture for Stationary Materials (SIGGRAPH 2015)</b> [[Paper]](https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/)
<p align="center"><img width="50%" src="https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/teaser.png" /></p>

<b>Reflectance Modeling by Neural Texture Synthesis (2016)</b> [[Paper]](https://mediatech.aalto.fi/publications/graphics/NeuralSVBRDF/)
<p align="center"><img width="50%" src="https://mediatech.aalto.fi/publications/graphics/NeuralSVBRDF/teaser.png" /></p>

<b>Modeling Surface Appearance from a Single Photograph using Self-augmented Convolutional Neural Networks (2017)</b> [[Paper]](http://msraig.info/~sanet/sanet.htm)
<p align="center"><img width="50%" src="http://msraig.info/~sanet/teaser.jpg" /></p>

<b>High-Resolution Multi-Scale Neural Texture Synthesis (2017)</b> [[Paper]](https://wxs.ca/research/multiscale-neural-synthesis/)
<p align="center"><img width="50%" src="https://wxs.ca/research/multiscale-neural-synthesis/multiscale-gram-marble.jpg" /></p>

<b>Reflectance and Natural Illumination from Single Material Specular Objects Using Deep Learning (2017)</b> [[Paper]](https://homes.cs.washington.edu/~krematas/Publications/reflectance-natural-illumination.pdf)
<p align="center"><img width="50%" src="http://www.vision.ee.ethz.ch/~georgous/images/tpami17_teaser2.png" /></p>

<b>Joint Material and Illumination Estimation from Photo Sets in the Wild (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.08313.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Joint%20Material%20and%20Illumination%20Estimation%20from%20Photo%20Sets%20in%20the%20Wild.jpeg" /></p>

<b>TextureGAN: Controlling Deep Image Synthesis with Texture Patches (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1706.02823.pdf)
<p align="center"><img width="50%" src="http://texturegan.eye.gatech.edu/img/paper_figure.png" /></p>

<b>Gaussian Material Synthesis (2018 SIGGRAPH)</b> [[Paper]](https://users.cg.tuwien.ac.at/zsolnai/gfx/gaussian-material-synthesis/)
<p align="center"><img width="50%" src="https://i.ytimg.com/vi/VM2ysCnD9GA/maxresdefault.jpg" /></p>

<b>Non-stationary Texture Synthesis by Adversarial Expansion (2018 SIGGRAPH)</b> [[Paper]](http://vcc.szu.edu.cn/research/2018/TexSyn)
<p align="center"><img width="50%" src="http://vcc.szu.edu.cn/upload/image/20180424/20180424130538_128.jpg" /></p>

<b>Synthesized Texture Quality Assessment via Multi-scale Spatial and Statistical Texture Attributes of Image and Gradient Magnitude Coefficients (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.08020.pdf)
<p align="center"><img width="50%" src="https://user-images.githubusercontent.com/12434910/39275366-e18c7c1c-4899-11e8-8e61-05072618bbce.PNG" /></p>

<b>LIME: Live Intrinsic Material Estimation (2018 CVPR)</b> [[Paper]](https://gvv.mpi-inf.mpg.de/projects/LIME/)
<p align="center"><img width="50%" src="https://web.stanford.edu/~zollhoef/papers/CVPR18_Material/teaser.png" /></p>

<a name="style_transfer" />

## Style Transfer
<b>Style-Content Separation by Anisotropic Part Scales (2010)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/xu_siga10_style.pdf)
<p align="center"><img width="50%" src="https://sites.google.com/site/kevinkaixu/_/rsrc/1472852123106/publications/style_b.jpg?height=145&width=400" /></p>

<b>Design Preserving Garment Transfer (2012)</b> [[Paper]](https://hal.inria.fr/hal-00695903/file/GarmentTransfer.pdf)
<p align="center"><img width="30%" src="https://hal.inria.fr/hal-00695903v2/file/02_WomanToAll.jpg" /></p>

<b>Analogy-Driven 3D Style Transfer (2014)</b> [[Paper]](http://www.chongyangma.com/publications/st/index.html)
<p align="center"><img width="50%" src="http://www.cs.ubc.ca/~chyma/publications/st/2014_st_teaser.png" /></p>

<b>Elements of Style: Learning Perceptual Shape Style Similarity (2015)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity.pdf) [[Code]](https://github.com/happylun/StyleSimilarity)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity_teaser.jpg" /></p>

<b>Functionality Preserving Shape Style Transfer (2016)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer.pdf) [[Code]](https://github.com/happylun/StyleTransfer)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer_teaser.jpg" /></p>

<b>Unsupervised Texture Transfer from Images to Model Collections (2016)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/siga16_texture_transfer_small.pdf)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2016/texture_transfer/paper_docs/teaser.png" /></p>

<b>Learning Detail Transfer based on Geometric Features (2017)</b> [[Paper]](http://surfacedetails.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://surfacedetails.cs.princeton.edu/images/teaser.png" /></p>

<b>Neural 3D Mesh Renderer (2017)</b> [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DPSm-4HWkAApEZd.jpg" /></p>

<b>Appearance Modeling via Proxy-to-Image Alignment (2018)</b> [[Paper]](http://vcc.szu.edu.cn/research/2018/AppMod)
<p align="center"><img width="50%" src="http://vcc.szu.edu.cn/viewFile/0/58/attached/image/20171026/20171026180502_864.jpg" /></p>

:gem: <b>Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (2018)</b> [[Paper]](http://bigvid.fudan.edu.cn/pixel2mesh/)
<p align="center"><img width="50%" src="http://bigvid.fudan.edu.cn/pixel2mesh/eccv2018/pipeline_01.jpg" /></p>

<a name="scene_synthesis" />

## Scene Synthesis/Reconstruction
<b>Make It Home: Automatic Optimization of Furniture Arrangement (2011, SIGGRAPH)</b> [[Paper]](http://people.sutd.edu.sg/~saikit/projects/furniture/index.html)
<p align="center"><img width="40%" src="https://www.cs.umb.edu/~craigyu/img/papers/furniture.gif" /></p>

<b>Interactive Furniture Layout Using Interior Design Guidelines (2011)</b> [[Paper]](http://graphics.stanford.edu/~pmerrell/furnitureLayout.htm)
<p align="center"><img width="50%" src="http://vis.berkeley.edu/papers/furnitureLayout/furnitureBig.jpg" /></p>

<b>Synthesizing Open Worlds with Constraints using Locally Annealed Reversible Jump MCMC (2012)</b> [[Paper]](http://graphics.stanford.edu/~lfyg/owl.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Synthesizing%20Open%20Worlds%20with%20Constraints%20using%20Locally%20Annealed%20Reversible%20Jump%20MCMC%20(2012).jpeg" /></p>

<b>Example-based Synthesis of 3D Object Arrangements (2012 SIGGRAPH Asia)</b> [[Paper]](http://graphics.stanford.edu/projects/scenesynth/)
<p align="center"><img width="60%" src="http://graphics.stanford.edu/projects/scenesynth/img/teaser.jpg" /></p>

<b>Sketch2Scene: Sketch-based Co-retrieval  and Co-placement of 3D Models  (2013)</b> [[Paper]](http://sweb.cityu.edu.hk/hongbofu/projects/sketch2scene_sig13/#.WWWge__ysb0)
<p align="center"><img width="40%" src="http://sunweilun.github.io/images/paper/sketch2scene_thumb.jpg" /></p>

<b>Action-Driven 3D Indoor Scene Evolution (2016)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/ma_siga16_action.pdf)
<p align="center"><img width="50%" src="https://maruitx.github.io/project/adise/teaser.jpg" /></p>

<b>Relationship Templates for Creating Scene Variations (2016)</b> [[Paper]](http://geometry.cs.ucl.ac.uk/projects/2016/relationship-templates/)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2016/relationship-templates/paper_docs/teaser.png" /></p>

<b>IM2CAD (2017)</b> [[Paper]](http://homes.cs.washington.edu/~izadinia/im2cad.html)
<p align="center"><img width="50%" src="http://i.imgur.com/KhtOeuB.jpg" /></p>

<b>Predicting Complete 3D Models of Indoor Scenes (2017)</b> [[Paper]](https://arxiv.org/pdf/1504.02437.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Predicting%20Complete%203D%20Models%20of%20Indoor%20Scenes.png" /></p>

<b>Complete 3D Scene Parsing from Single RGBD Image (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.09490.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Complete%203D%20Scene%20Parsing%20from%20Single%20RGBD%20Image.jpeg" /></p>

<b>Raster-to-Vector: Revisiting Floorplan Transformation (2017, ICCV)</b> [[Paper]](http://www.cse.wustl.edu/~chenliu/floorplan-transformation.html) [[Code]](https://github.com/art-programmer/FloorplanTransformation)
<p align="center"><img width="50%" src="https://www.cse.wustl.edu/~chenliu/floorplan-transformation/teaser.png" /></p>

<b>Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes (2017)</b> [[Blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)
<p align="center"><img width="50%" src="https://cdn-images-1.medium.com/max/1600/1*NckW2hfgbHhEP3P8Z5ZLjQ.png" /></p>

<b>Adaptive Synthesis of Indoor Scenes via Activity-Associated Object Relation Graphs (2017 SIGGRAPH Asia)</b> [[Paper]](http://arts.buaa.edu.cn/projects/sa17/)
<p align="center"><img width="50%" src="https://sa2017.siggraph.org/images/events/c121-e45-publicimage.jpg" /></p>

<b>Human-centric Indoor Scene Synthesis Using Stochastic Grammar (2018, CVPR)</b>[[Paper]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis.pdf) [[Supplementary]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis_supplementary.pdf) [[Code]](https://github.com/SiyuanQi/human-centric-scene-synthesis)
<p align="center"><img width="50%" src="http://web.cs.ucla.edu/~syqi/publications/thumbnails/cvpr2018synthesis.gif" /></p>

:camera::game_die: <b>FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1804.00090.pdf) [[Code]](http://art-programmer.github.io/floornet.html)
<p align="center"><img width="50%" src="http://art-programmer.github.io/floornet/teaser.png" /></p>

:space_invader: <b>ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.10215.pdf) 
<p align="center"><img width="50%" src="https://cs.stanford.edu/~adai/papers/2018/0scancomplete/teaser.jpg" /></p>

<b>Deep Convolutional Priors for Indoor Scene Synthesis (2018)</b> [[Paper]](https://kwang-ether.github.io/pdf/deepsynth.pdf) 
<p align="center"><img width="50%" src="http://msavva.github.io/files/deepsynth.png" /></p>

<a name="scene_understanding" />

## Scene Understanding
<b>Characterizing Structural Relationships in Scenes Using Graph Kernels (2011 SIGGRAPH)</b> [[Paper]](https://graphics.stanford.edu/~mdfisher/graphKernel.html)
<p align="center"><img width="60%" src="https://graphics.stanford.edu/~mdfisher/papers/graphKernelTeaser.png" /></p>

<b>Understanding Indoor Scenes Using 3D Geometric Phrases (2013)</b> [[Paper]](http://cvgl.stanford.edu/projects/3dgp/)
<p align="center"><img width="30%" src="http://cvgl.stanford.edu/projects/3dgp/images/title.png" /></p>

<b>Organizing Heterogeneous Scene Collections through Contextual Focal Points (2014 SIGGRAPH)</b> [[Paper]](http://kevinkaixu.net/projects/focal.html)
<p align="center"><img width="60%" src="http://kevinkaixu.net/projects/focal/overlapping_clusters.jpg" /></p>

<b>SceneGrok: Inferring Action Maps in 3D Environments (2014, SIGGRAPH)</b> [[Paper]](http://graphics.stanford.edu/projects/scenegrok/)
<p align="center"><img width="50%" src="http://graphics.stanford.edu/projects/scenegrok/scenegrok.png" /></p>

<b>PanoContext: A Whole-room 3D Context Model for Panoramic Scene Understanding (2014)</b> [[Paper]](http://panocontext.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://panocontext.cs.princeton.edu/teaser.jpg" /></p>

<b>Learning Informative Edge Maps for Indoor Scene Layout Prediction (2015)</b> [[Paper]](http://web.engr.illinois.edu/~slazebni/publications/iccv15_informative.pdf)
<p align="center"><img width="50%" src="http://img.blog.csdn.net/20170820185439268?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYmFpeXU5ODIxMTc5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" /></p>

<b>Rent3D: Floor-Plan Priors for Monocular Layout Estimation (2015)</b> [[Paper]](http://www.cs.toronto.edu/~fidler/projects/rent3D.html)
<p align="center"><img width="50%" src="http://www.cs.toronto.edu/~fidler/projects/layout-res.jpg" /></p>

<b>A Coarse-to-Fine Indoor Layout Estimation (CFILE) Method (2016)</b> [[Paper]](https://pdfs.semanticscholar.org/7024/a92186b81e6133dc779f497d06877b48d82b.pdf?_ga=2.54181869.497995160.1510977308-665742395.1510465328)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/A%20Coarse-to-Fine%20Indoor%20Layout%20Estimation%20(CFILE)%20Method%20(2016).png" /></p>

<b>DeLay: Robust Spatial Layout Estimation for Cluttered Indoor Scenes (2016)</b> [[Paper]](http://deeplayout.stanford.edu/)
<p align="center"><img width="30%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/DeLay-Robust%20Spatial%20Layout%20Estimation%20for%20Cluttered%20Indoor%20Scenes.png" /></p>

<b>3D Semantic Parsing of Large-Scale Indoor Spaces (2016)</b> [[Paper]](http://buildingparser.stanford.edu/method.html) [[Code]](https://github.com/alexsax/2D-3D-Semantics)
<p align="center"><img width="50%" src="http://buildingparser.stanford.edu/images/teaser.png" /></p>

<b>Single Image 3D Interpreter Network (2016)</b> [[Paper]](http://3dinterpreter.csail.mit.edu/) [[Code]](https://github.com/jiajunwu/3dinn)
<p align="center"><img width="50%" src="http://3dinterpreter.csail.mit.edu/images/spotlight_3dinn_large.jpg" /></p>

<b>Deep Multi-Modal Image Correspondence Learning (2016)</b> [[Paper]](http://www.cse.wustl.edu/~chenliu/floorplan-matching.html)
<p align="center"><img width="50%" src="https://www.cse.wustl.edu/~furukawa/carousel/2016-floorplan-wide.png" /></p>

<b>Physically-Based Rendering for Indoor Scene Understanding Using Convolutional Neural Networks (2017)</b> [[Paper]](http://3dvision.princeton.edu/projects/2016/PBRS/) [[Code]](https://github.com/yindaz/pbrs) [[Code]](https://github.com/yindaz/surface_normal) [[Code]](https://github.com/fyu/dilation) [[Code]](https://github.com/s9xie/hed)
<p align="center"><img width="50%" src="http://robots.princeton.edu/projects/2016/PBRS/teaser.jpg" /></p>

<b>RoomNet: End-to-End Room Layout Estimation (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.06241.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/C7Z29GsV0AASEvR.jpg" /></p>

<b>SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite (2017)</b> [[Paper]](http://rgbd.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://rgbd.cs.princeton.edu/teaser.jpg" /></p>

<b>Semantic Scene Completion from a Single Depth Image (2017)</b> [[Paper]](http://sscnet.cs.princeton.edu/) [[Code]](https://github.com/shurans/sscnet)
<p align="center"><img width="50%" src="http://sscnet.cs.princeton.edu/teaser.jpg" /></p>

<b>Factoring Shape, Pose, and Layout  from the 2D Image of a 3D Scene (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1712.01812.pdf) [[Code]](https://shubhtuls.github.io/factored3d/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/factored3d/resources/images/teaser.png" /></p>

<b>LayoutNet: Reconstructing the 3D Room Layout from a Single RGB Image (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1803.08999.pdf) [[Code]](https://github.com/zouchuhang/LayoutNet)
<p align="center"><img width="50%" src="http://p0.ifengimg.com/pmop/2018/0404/A1D0CAE48130C918FE624FA60495F237C67172F6_size63_w797_h755.jpeg" /></p>

<b>PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image (2018 CVPR)</b> [[Paper]](http://art-programmer.github.io/planenet/paper.pdf) [[Code]](http://art-programmer.github.io/planenet.html)
<p align="center"><img width="50%" src="http://art-programmer.github.io/images/planenet.png" /></p>

<b>Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery (2018 CVPR)</b> [[Paper]](http://web.cs.ucdavis.edu/~yjlee/projects/cvpr2018.pdf) <p align="center"><img width="50%" src="https://jason718.github.io/project/cvpr18/files/concept_pic.png" /></p>

<b>Pano2CAD: Room Layout From A Single Panorama Image (2018 CVPR)</b> [[Paper]](http://bjornstenger.github.io/papers/xu_wacv2017.pdf) <p align="center"><img width="50%" src="https://www.groundai.com/media/arxiv_projects/58924/figures/Compare_2b.png" /></p>

<b>Automatic 3D Indoor Scene Modeling from Single Panorama (2018 CVPR)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Automatic_3D_Indoor_CVPR_2018_paper.pdf) <p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Automatic%203D%20Indoor%20Scene%20Modeling%20from%20Single%20Panorama%20(2018%20CVPR).jpeg" /></p>

