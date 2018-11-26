# Datasets
To see a survey of RGBD datasets, check out Michael Firman's [collection](http://www0.cs.ucl.ac.uk/staff/M.Firman//RGBDdatasets/) as well as the associated paper, [RGBD Datasets: Past, Present and Future](https://arxiv.org/pdf/1604.00999.pdf). Point Cloud Library also has a good dataset [catalogue](http://pointclouds.org/media/). 

<a name="3d_models" />

## 3D Models
<b>Princeton Shape Benchmark (2003)</b> [[Link]](http://shape.cs.princeton.edu/benchmark/)
<br>1,814 models collected from the web in .OFF format. Used to evaluating shape-based retrieval and analysis algorithms.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Princeton%20Shape%20Benchmark%20(2003).jpeg" /></p>

<b>Dataset for IKEA 3D models and aligned images (2013)</b> [[Link]](http://ikea.csail.mit.edu/)
<br>759 images and 219 models including Sketchup (skp) and Wavefront (obj) files, good for pose estimation.
<p align="center"><img width="50%" src="http://ikea.csail.mit.edu/web_img/ikea_object.png" /></p>

<b>PASCAL3D+ (2014)</b> [[Link]](http://cvgl.stanford.edu/projects/pascal3d.html)
<br>12 categories, on average 3k+ objects per category, for 3D object detection and pose estimation.
<p align="center"><img width="50%" src="http://cvgl.stanford.edu/projects/pascal3d+/pascal3d.png" /></p>

<b>SHREC 2014 - Large Scale Comprehensive 3D Shape Retrieval (2014)</b> [[Link]](http://www.itl.nist.gov/iad/vug/sharp/contest/2014/Generic3D/index.html)[[Paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.431.7090&rep=rep1&type=pdf)
<br>8,987 models, categorized into 171 classes for shape retrieval.
<p align="center"><img width="50%" src="http://www.itl.nist.gov/iad/vug/sharp/contest/2014/Generic3D/index_files/fig1.png" /></p>

<b>ModelNet (2015)</b> [[Link]](http://modelnet.cs.princeton.edu/#)
<br>127915 3D CAD models from 662 categories
<br>ModelNet10: 4899 models from 10 categories
<br>ModelNet40: 12311 models from 40 categories, all are uniformly orientated
<p align="center"><img width="50%" src="http://3dvision.princeton.edu/projects/2014/ModelNet/thumbnail.jpg" /></p>

<b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.
<br>ShapeNetCore [[Link]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
<p align="center"><img width="50%" src="http://msavva.github.io/files/shapenet.png" /></p>

<b>A Large Dataset of Object Scans (2016)</b> [[Link]](http://redwood-data.org/3dscan/index.html)
<br>10K scans in RGBD + reconstructed 3D models in .PLY format.
<p align="center"><img width="50%" src="http://redwood-data.org/3dscan/img/teaser.jpg" /></p>

<b>ObjectNet3D: A Large Scale Database for 3D Object Recognition (2016)</b> [[Link]](http://cvgl.stanford.edu/projects/objectnet3d/)
<br>100 categories, 90,127 images, 201,888 objects in these images and 44,147 3D shapes. 
<br>Tasks: region proposal generation, 2D object detection, joint 2D detection and 3D object pose estimation, and image-based 3D shape retrieval
<p align="center"><img width="50%" src="http://cvgl.stanford.edu/projects/objectnet3d/ObjectNet3D.png" /></p>

<b>Thingi10K: A Dataset of 10,000 3D-Printing Models (2016)</b> [[Link]](https://ten-thousand-models.appspot.com/)
<br>10,000 models from featured “things” on thingiverse.com, suitable for testing 3D printing techniques such as structural analysis , shape optimization, or solid geometry operations.
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DRbxWnqXkAEEH0g.jpg:large" /></p>

<a name="3d_scenes" />

## 3D Scenes
<b>NYU Depth Dataset V2 (2012)</b> [[Link]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
<br>1449 densely labeled pairs of aligned RGB and depth images from Kinect video sequences for a variety of indoor scenes.
<p align="center"><img width="50%" src="https://cs.nyu.edu/~silberman/images/nyu_depth_v2_labeled.jpg" /></p>

<b>SUNRGB-D 3D Object Detection Challenge</b> [[Link]](http://rgbd.cs.princeton.edu/challenge.html)
<br>19 object categories for predicting a 3D bounding box in real world dimension
<br>Training set: 10,355 RGB-D scene images, Testing set: 2860 RGB-D images
<p align="center"><img width="50%" src="http://rgbd.cs.princeton.edu/3dbox.png" /></p>

<b>SceneNN (2016)</b> [[Link]](http://people.sutd.edu.sg/~saikit/projects/sceneNN/)
<br>100+ indoor scene meshes with per-vertex and per-pixel annotation.
<p align="center"><img width="50%" src="https://cdn-ak.f.st-hatena.com/images/fotolife/r/robonchu/20170611/20170611155625.png" /></p>

<b>ScanNet (2017)</b> [[Link]](http://www.scan-net.org/)
<br>An RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.
<p align="center"><img width="50%" src="http://www.scan-net.org/img/annotations.png" /></p>

<b>Matterport3D: Learning from RGB-D Data in Indoor Environments (2017)</b> [[Link]](https://niessner.github.io/Matterport/)
<br>10,800 panoramic views (in both RGB and depth) from 194,400 RGB-D images of 90 building-scale scenes of private rooms. Instance-level semantic segmentations are provided for region (living room, kitchen) and object (sofa, TV) categories. 
<p align="center"><img width="50%" src="https://niessner.github.io/Matterport/teaser.png" /></p>

<b>SUNCG: A Large 3D Model Repository for Indoor Scenes (2017)</b> [[Link]](http://suncg.cs.princeton.edu/)
<br>The dataset contains over 45K different scenes with manually created realistic room and furniture layouts. All of the scenes are semantically annotated at the object level.
<p align="center"><img width="50%" src="http://suncg.cs.princeton.edu/figures/data_full.png" /></p>

<b>MINOS: Multimodal Indoor Simulator (2017)</b> [[Link]](https://github.com/minosworld/minos)
<br>MINOS is a simulator designed to support the development of multisensory models for goal-directed navigation in complex indoor environments. MINOS leverages large datasets of complex 3D environments and supports flexible configuration of multimodal sensor suites. MINOS supports SUNCG and Matterport3D scenes.
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2017/12/MINOS.jpg" /></p>

<b>Facebook House3D: A Rich and Realistic 3D Environment (2017)</b> [[Link]](https://github.com/facebookresearch/House3D)
<br>House3D is a virtual 3D environment which consists of 45K indoor scenes equipped with a diverse set of scene types, layouts and objects sourced from the SUNCG dataset. All 3D objects are fully annotated with category labels. Agents in the environment have access to observations of multiple modalities, including RGB images, depth, segmentation masks and top-down 2D map views.
<p align="center"><img width="50%" src="https://user-images.githubusercontent.com/1381301/33509559-87c4e470-d6b7-11e7-8266-27c940d5729a.jpg" /></p>

<b>HoME: a Household Multimodal Environment (2017)</b> [[Link]](https://home-platform.github.io/)
<br>HoME integrates over 45,000 diverse 3D house layouts based on the SUNCG dataset, a scale which may facilitate learning, generalization, and transfer. HoME is an open-source, OpenAI Gym-compatible platform extensible to tasks in reinforcement learning, language grounding, sound-based navigation, robotics, multi-agent learning.
<p align="center"><img width="50%" src="https://home-platform.github.io/assets/overview.png" /></p>

<b>AI2-THOR: Photorealistic Interactive Environments for AI Agents</b> [[Link]](http://ai2thor.allenai.org/)
<br>AI2-THOR is a photo-realistic interactable framework for AI agents. There are a total 120 scenes in version 1.0 of the THOR environment covering four different room categories: kitchens, living rooms, bedrooms, and bathrooms. Each room has a number of actionable objects.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/AI2-Thor.jpeg" /></p>

<b>UnrealCV: Virtual Worlds for Computer Vision (2017)</b> [[Link]](http://unrealcv.org/)[[Paper]](http://www.idm.pku.edu.cn/staff/wangyizhou/papers/ACMMM2017_UnrealCV.pdf)
<br>An open source project to help computer vision researchers build virtual worlds using Unreal Engine 4.
<p align="center"><img width="50%" src="http://unrealcv.org/images/homepage_teaser.png" /></p>

<b>Gibson Environment: Real-World Perception for Embodied Agents (2018 CVPR) </b> [[Link]](http://gibsonenv.stanford.edu/)
<br>This platform provides RGB from 1000 point clouds, as well as multimodal sensor data: surface normal, depth, and for a fraction of the spaces, semantics object annotations. The environment is also RL ready with physics integrated. Using such datasets can further narrow down the discrepency between virtual environment and real world.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Gibson%20Environment-%20Real-World%20Perception%20for%20Embodied%20Agents%20(2018%20CVPR)%20.jpeg" /></p>