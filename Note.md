## TODO
- 将现有的双目ORB+IMU跑起来
- 前端光流
- 后端的sliding window
- Consider add inv depth prior in estimation (done.)
- 当Point的Ref出窗时，应该换ref还是marg还是固定？
- 新关键帧策略，NeedNewKeyFrame()改成类似ORB
- check "v is nullptr in Local ba without imu"
- 多线程

## 日志

### 6.6
- 把shared_ptr相关修改合并到master
- EdgeProjectPoseOnly的computeError()里，如果invz<0，不能把_error置零，否则会当成在poseoptimization中会当成inlier
- 测试纯视觉的方案，见test/testPureVision.cpp

### 6.5
- 把很多内存相关的东西都改成了shared_ptr和weak_ptr，再也不用担心内存泄漏啦！

### 6.4
- wj:
- Frame::SetDelete在修改完shared_ptr之后，要根据情况再考虑

### 6.3
- xiang:
- 16.04或同版本linux下，g2o/core/jacobian_workspace.cpp里有一句setZero，需要加上维度。否则易导致在开辟雅可比空间时seg fault.

### 6.2
- fix many Eigen::aligned_allocator problem
- rewrite the test program in test/xxx
- add stereo imu initialization code
- wj:
- 在imuinitialization中增加g=9.81的约束，加一步估计步骤。
- 在后端纯视觉的LocalBA中加入：对outlier的观测进行删除
- imuinitialization的成功条件：无约束估计出的g的模长在9.6~10.0之间，并且约束g=9.81前后估计出加速度计零偏的误差模长小于0.2m/s^2
  注：看log拍脑袋想的条件。。TBD
- 在V101中，31个KF左右可以满足条件（从第6s开始。前6s没动，测试时被跳过），MH01要50多个KF。估计出的零偏可能还有误差，等后续继续优化。
- imuinitialization成功后，将w系与重力方向对齐，使mgWrold=[0,0,9.81]，这样可以认为这个重力是真值，不需要对它进行优化更新。


### 5.31
- 修改了testViewer里的内存管理问题。 

### 5.30 
- 增加了后端新增地图点的过程，但未测试

### 5.29
- fix many things in tracker and ORBExtractor, ORBMatcher by testing stereo init. 


### 5.27
- 增加了Viewer的测试，见test/testViewer.cpp
- 由于pangolin/OpenGL的问题，在osx下新开openGL线程时会导致出错，所以testViewer在主线程中调用可视化

### 5.21
- 准备开始测试带IMU的双目初始化部分代码 

### 5.20 
- 增加了带IMU的Local BA测试，见test/testLocalBAIMU
- 在osx下，g2o的optimizer在析构，删除顶点时会产生double free问题，原因不明。Ubuntu下没有问题。

### 5.17
- 后端LocalBA通过测试，见test/testLocalBA

### 5.12
- 添加了后端的两个BA，待测试

### 5.10
- 添加一些g2otypes，使用P+R,V,Ba,Bg进行基本的表示 
- 开始添加Tracker内容

### 4.27 数据结构基本完成，加入特征提取（待测试）
### 2017.4.26 调整架构，加入前端算法
