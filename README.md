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

