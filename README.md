**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Keyu Lu
* Tested on: Windows 10, Dell Oman, NVIDIA GeForce RTX 2060


### Flocking Results
| Flocking Scene: N_FOR_VIS = 5,000, scene_scale=100.0f, DT = 0.2f | 
|---------------|
| ![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/flocking%20demo.gif) | 

| Flocking Scene: N_FOR_VIS = 500,000, scene_scale=300.0f, DT = 2.0f | 
|---------------|
| ![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/500k%20300%202.0%20demo%20gif.gif) | 

| Flocking Scene: N_FOR_VIS = 5000,000, scene_scale=500.0f, DT = 0.5f | 
|---------------|
| ![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/5m%20500%200.5.gif) | 

### Performance Analysis 
#### Boids Count Impact on Performance

![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/Numbers%20of%20Boids%20VS.%20FPS.png)
![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/Numbers%20of%20Boids%20VS.%20FPS%20(with%20Visualization).png)

**Naive:** Performance drops sharply as boid count increases due to O(N^2) complexity.
**Uniform Grid:** Less severe performance drop with more boids due to reduced comparisons.
**Coherent Grid:** Similar to the uniform grid but maintains better performance due to optimized memory access.

![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/Framerate%20Change%20with%20Increasing%20Block%20Size.png)

#### Block Size and Block Count Effects
Increasing block size generally improves performance until a threshold, after which there are diminishing returns. This is consistent across all implementations and is likely due to the limits of GPU thread management and optimal thread occupancy.

#### Cell Width and Neighbor Checking
**27 vs. 8 Neighboring Cells:** 

## Extra Credit: Shared-Memory Optimization

### Implementation

For the extra credit, I implemented a shared-memory optimization to enhance the nearest neighbor search within the naive approach of the boid simulation. The naive approach's performance was improved by using shared memory for the computations involved in updating boid velocities.

The implementation conditional can be observed in the following snippet:

```cpp
#if USE_SHARED_MEM
kernUpdateVelocityBruteForceShared <<< fullBlocksPerGrid, blockSize, sizeof(glm::vec3) * blockSize * 2 >>> (N, dev_pos, dev_vel1, dev_vel2);
#else
kernUpdateVelocityBruteForce <<< fullBlocksPerGrid, blockSize >>> (N, dev_pos, dev_vel1, dev_vel2);
#endif
```

This section of the code utilizes the preprocessor directive USE_SHARED_MEM to switch between using shared memory (kernUpdateVelocityBruteForceShared) and not using it (kernUpdateVelocityBruteForce).
#### Performance Analysis
As demonstrated, the use of shared memory has a significant impact on the frames per second (FPS) achieved by the simulation under the naive setting:
![](https://github.com/uluyek/Project1-CUDA-Flocking/blob/main/FPS%20for%20Naive%20Implementation%20with%20Shared%20Memory%20On_Off.png)
