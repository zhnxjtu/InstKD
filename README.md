# InstKD
This is part of the key Pytorch implementation (the complete project will be available) of our paper "InstKD: Towards Lightweight 3D Object Detection with Instance-aware Knowledge Distillation". Our codes and pre-trained teacher models are based on OpenPCDet and SparseKD detection framework. Thanks OpenPCDet Development Team for their awesome codebase (The complete code will be available).

1. Pre-trained Teacher model: The pre-trained teacher models and raw student models are trained by running '.yaml' file in OpenPCDet.

2. InstKD method: In './Code_InstKD/pcdet/models/kd_trans_block', we provide the main code of our distillation method. 
Download SparseKD detection framework, and then move './Code_InstKD/pcdet/models/kd_trans_block' to './SparseKD/pcdet/models/kd_trans_block'. Also, a 'build_kd_trans_block' function should be added in './SparseKD/pcdet/models/detectors/detector3d_template.py' to build knowledge transfer block in the training phase.

3. KD configs: In './Code_InstKD/tools/cfgs', we also provide the configuration files.

We will release all the codes in github after reviewing.
